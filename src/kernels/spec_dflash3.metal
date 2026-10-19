#include "common.h"

// ---- DFlash3Turbo candidate decoder (src/model/dflash3.cpp; internal/DFLASH3TURBO.md §1, §4; format and reference
// forward: tools/dflash3/README.md, model.py of the trainer) ----
// A small autoregressive decoder over the drafted path replaces DFlash2's first-order selector. The drafting tree
// lives on the GPU (D3Tree, shared with the tree builder): node 0 is the anchor (depth 0); a node at depth j is a draft
// at block row j. Expanding a node n at depth j runs the decoder's path row j over its branch (root .. n) and scores
// n's children, the backbone's top-32 candidates C_{j+1} of block row j + 1:
//   x   = W_h h_{j+1} + W_e (RMSNorm(E[tok(n)]) * emb_norm) + pos[j]   (h = the backbone's final normed hidden, E = the
//                                                                      target's embedding; W_h h once per cycle)
//   two pre-norm layers: multi-head attention 8 x 128 over the branch rows 0 .. j (no RoPE), SwiGLU MLP
//   y   = RMSNorm(.; norm),  z = W_out y   (d -> H)
//   S[k] = logit_scale[j] U_{j+1}[k] + <z, E_out[C_{j+1}[k]]>   (U = draft logits, E_out = the target's LM head)
//   acceptance row r (0: greedy, 1: sampled): sigmoid(<w_y[r], y> + <w_e[r], E_out[c] / rms(E_out[c])>
//                                                     + w_q[r] log softmax(S)[k] + b[r])
// One decoder step expands up to 8 front nodes as the rows of the drafter's 8-row GEMMs (dfl_qmm8, dfl_qmm8_gu); the
// node K/V rows stay in kv [64 nodes][layers][K | V][d] (fp32), written when the node is expanded and read by its
// descendants. A front may hold a node together with some of its descendants (a speculated chain or subtree whose
// tokens the builder picked from the candidates before their scores exist): the attention then takes those
// ancestors' K/V from the step's own q|k|v rows (tree-causal attention), so every row is still conditioned on exactly
// its branch. Every ancestor of a front node must be expanded already or be in the same front.
// The file sorts after spec_dflash.metal in the embedded library (dfl_put2, dfl_uniform, dfl_pos, kDflSaltWalk).

constant constexpr uint kD3Cand = 32;
constant constexpr uint kD3MaxNodes = 64;
constant constexpr uint kD3MaxFront = 8;
constant constexpr uint kD3MaxDepth = 16;

// The tree buffer (src/model/dflash3.h D3Tree, same layout). Children arrays are indexed [node * 32 + k].
struct D3Tree {
  uint n_nodes;               // nodes in the tree (node 0 = the anchor)
  uint n_front;               // nodes the next decoder step expands (<= 8)
  uint front[kD3MaxFront];    // their ids
  uint P;                     // anchor position
  uint walk_at;               // the speculative chain walk (d3_jac): the chain node whose row the next pass decides
  uint walk_done;             // 1 once the chain is complete
  uint walk_passes;           // passes that decided rows
  uint spare[2];
  int parent[kD3MaxNodes];    // -1 for the root
  int token[kD3MaxNodes];
  int depth[kD3MaxNodes];
  int cand_k[kD3MaxNodes];    // index among the parent's candidates (-1 for the root)
  float logp[kD3MaxNodes];    // log path probability (sum of log softmax(S) along the root path)
  float acc[kD3MaxNodes];     // acceptance-head probability of the node given its parent's path
  int expanded[kD3MaxNodes];  // 1 once a decoder step scored its children
  int row[kD3MaxNodes];       // verify row (the tree builder's)
  int ch_tok[kD3MaxNodes * kD3Cand];
  float ch_s[kD3MaxNodes * kD3Cand];  // raw scores S
  float ch_q[kD3MaxNodes * kD3Cand];  // softmax(S / T_q)
  float ch_a[kD3MaxNodes * kD3Cand];  // acceptance head
};

// Elements k, k + 1 (k even) of row `row` of an affine bits-bit (4 or 8), group-gs table with K columns, in fp32
// (scale * code + bias, as dfl_embed_norm).
NI_INLINE float2 d3_row2(const device uint32_t* W, const device bfloat* S, const device bfloat* B, uint row, uint k,
                         uint K, uint bits, uint gs) {
  const uint gpr = K / gs;
  const float sc = float(S[size_t(row) * gpr + k / gs]), bi = float(B[size_t(row) * gpr + k / gs]);
  if (bits == 4) {
    const uint word = W[size_t(row) * (K / 8) + k / 8], sh = 4 * (k % 8);
    return float2(sc * float((word >> sh) & 15u) + bi, sc * float((word >> (sh + 4)) & 15u) + bi);
  }
  const uint word = W[size_t(row) * (K / 4) + k / 4], sh = 8 * (k % 4);
  return float2(sc * float((word >> sh) & 255u) + bi, sc * float((word >> (sh + 8)) & 255u) + bi);
}

// Root of a drafting tree: node 0 = the anchor tokens[P] (P == kDflGpu: the drafter state's anchor), front = {0}.
// spec > 0 (the speculative chain walk): nodes 1 .. spec below it, each the backbone's top-1 candidate of its row
// (cand_i[(j - 1) * 32]), all in the front: the first pass scores that whole chain. grid 1 x 32.
struct D3RootParams {
  uint P, spec;
};
[[kernel]] void d3_root(device D3Tree* t [[buffer(0)]], const device int* tokens [[buffer(1)]],
                        constant D3RootParams& rp [[buffer(2)]], const device uint* gst [[buffer(3)]],
                        const device int* cand_i [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const uint p = dfl_pos(rp.P, gst);
  t->n_nodes = 1 + rp.spec;
  t->n_front = 1 + rp.spec;
  t->front[0] = 0;
  t->P = p;
  t->walk_at = 0;
  t->walk_done = 0;
  t->walk_passes = 0;
  for (uint j = 1; j <= rp.spec; ++j) {
    t->front[j] = j;
    t->parent[j] = int(j - 1);
    t->token[j] = cand_i[(j - 1) * kD3Cand];
    t->depth[j] = int(j);
    t->cand_k[j] = 0;
    t->logp[j] = 0.0f;
    t->acc[j] = 0.0f;
    t->expanded[j] = 0;
    t->row[j] = int(j);
  }
  t->parent[0] = -1;
  t->token[0] = tokens[p];
  t->depth[0] = 0;
  t->cand_k[0] = -1;
  t->logp[0] = 0.0f;
  t->acc[0] = 1.0f;
  t->expanded[0] = 0;
  t->row[0] = 0;
}

// Decoder input of the front rows: the X8 table (K = H) of RMSNorm(E[tok]) * emb_norm and the residual stream's
// start A[dep + 1] + pos[dep] (A = W_h h of the cycle's block rows, fp32 [8][d]); rows past n_front are zero.
// grid 8 x 256.
struct D3InParams {
  uint H, d, emb_bits, emb_gs;
  float eps;
};
[[kernel]] void d3_in(const device D3Tree* t [[buffer(0)]], const device float* A [[buffer(1)]],
                      const device uint32_t* EW [[buffer(2)]], const device bfloat* ES [[buffer(3)]],
                      const device bfloat* EB [[buffer(4)]], const device bfloat* en [[buffer(5)]],
                      const device bfloat* pos [[buffer(6)]], device half* xh [[buffer(7)]],
                      device float* xs [[buffer(8)]], device float* hres [[buffer(9)]],
                      constant D3InParams& p [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  const uint H = p.H, d = p.d;
  const bool live = r < t->n_front;
  const uint node = live ? t->front[r] : 0u;
  const uint dep = live ? uint(t->depth[node]) : 0u;
  const uint tok = live ? uint(t->token[node]) : 0u;
  float ss = 0.0f;
  if (live)
    for (uint g = sg; g < H / 64; g += nsg) {
      const float2 v = d3_row2(EW, ES, EB, tok, g * 64 + 2 * lane, H, p.emb_bits, p.emb_gs);
      ss += v.x * v.x + v.y * v.y;
    }
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(H) + p.eps);
  for (uint g = sg; g < H / 64; g += nsg) {
    const uint k = g * 64 + 2 * lane;
    float2 v = float2(0.0f);
    if (live) v = d3_row2(EW, ES, EB, tok, k, H, p.emb_bits, p.emb_gs) * inv * float2(float(en[k]), float(en[k + 1]));
    dfl_put2(xh, xs, r, H, g, lane, v.x, v.y);
  }
  for (uint c = tid; c < d; c += ntg)
    hres[r * d + c] = live ? A[(dep + 1) * d + c] + float(pos[dep * d + c]) : 0.0f;
}

// Attention of the front rows over their branches, one threadgroup of 64 per (row, head): this thread owns dims
// c, c + 1 of the head. The row's own K/V (from the q|k|v GEMM) go to the node's kv slot for its descendants; keys
// are the branch rows 0 .. j: an ancestor's from its kv slot, or from the q|k|v rows when it is in this front too.
// The output -> X8 table (the o projection's input); rows past n_front write zeros. grid (8 rows, heads) x 64.
struct D3AttnParams {
  uint d, layer, nl;
  float scale;
};
[[kernel]] void d3_attn(const device D3Tree* t [[buffer(0)]], const device float* qkv [[buffer(1)]],
                        device float* kv [[buffer(2)]], device half* xh [[buffer(3)]], device float* xs [[buffer(4)]],
                        constant D3AttnParams& p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2][kD3MaxDepth];
  const uint f = tg.x, hd = tg.y, d = p.d;
  const uint c = hd * 128 + sg * 64 + 2 * lane;
  if (f >= t->n_front) {
    dfl_put2(xh, xs, f, d, 2 * hd + sg, lane, 0.0f, 0.0f);
    return;
  }
  const uint node = t->front[f];
  const uint dep = min(uint(t->depth[node]), kD3MaxDepth - 1);
  // the branch's K / V rows: kv slots of expanded ancestors, this step's q|k|v rows for the ones in the front
  const size_t kvn = size_t(2) * d;  // floats per node and layer
  const uint nf = t->n_front;
  const device float* kr[kD3MaxDepth];
  {
    int a = int(node);
    for (int j = int(dep); j >= 0; --j) {
      uint slot = kD3MaxFront;
      for (uint s2 = 0; s2 < nf; ++s2)
        if (t->front[s2] == uint(a)) slot = s2;
      kr[j] = slot < kD3MaxFront ? qkv + size_t(slot) * 3 * d + d : kv + (size_t(a) * p.nl + p.layer) * kvn;
      a = t->parent[a];
    }
  }
  const device float* row = qkv + size_t(f) * 3 * d;
  const float2 q = *(const device float2*)(row + c);
  const float2 kown = *(const device float2*)(row + d + c), vown = *(const device float2*)(row + 2 * d + c);
  device float* mine = kv + (size_t(node) * p.nl + p.layer) * kvn;
  *(device float2*)(mine + c) = kown;
  *(device float2*)(mine + d + c) = vown;
  float s[kD3MaxDepth];
  for (uint j = 0; j <= dep; ++j) {
    const float2 k = j == dep ? kown : *(const device float2*)(kr[j] + c);
    s[j] = simd_sum(q.x * k.x + q.y * k.y);
  }
  if (lane == 0)
    for (uint j = 0; j <= dep; ++j) part[sg][j] = s[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mx = -INFINITY;
  for (uint j = 0; j <= dep; ++j) {
    s[j] = (part[0][j] + part[1][j]) * p.scale;
    mx = max(mx, s[j]);
  }
  float tot = 0.0f;
  for (uint j = 0; j <= dep; ++j) {
    s[j] = metal::precise::exp(s[j] - mx);
    tot += s[j];
  }
  float2 o = float2(0.0f);
  for (uint j = 0; j <= dep; ++j) {
    const float2 v = j == dep ? vown : *(const device float2*)(kr[j] + d + c);  // V follows K in both layouts
    o += s[j] * v;
  }
  o /= tot;
  dfl_put2(xh, xs, f, d, 2 * hd + sg, lane, o.x, o.y);
}

// Candidate dot products of the front rows, one threadgroup per (row f, candidate k): the LM-head row e = E_out[c]
// (c = C_{dep+1}[k]) against z_f and the acceptance head's two w_e rows, and its sum of squares ->
// cd[f][k] = (<z, e>, <w_e[0], e>, <w_e[1], e>, sum e^2). grid (8 rows, 32 candidates) x 256.
struct D3CandParams {
  uint H, out_bits;
};
[[kernel]] void d3_cand(const device D3Tree* t [[buffer(0)]], const device float* z [[buffer(1)]],
                        const device int* cand_i [[buffer(2)]], const device uint32_t* OW [[buffer(3)]],
                        const device bfloat* OS [[buffer(4)]], const device bfloat* OB [[buffer(5)]],
                        const device float* we [[buffer(6)]], device float4* cd [[buffer(7)]],
                        constant D3CandParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 red[32];
  const uint f = tg.x, k = tg.y, H = p.H, ntg = nsg * 32;
  if (f >= t->n_front) return;
  const uint node = t->front[f];
  const uint c = uint(cand_i[uint(t->depth[node]) * kD3Cand + k]);
  const device float* zr = z + size_t(f) * H;
  float4 acc = float4(0.0f);
  for (uint i = 2 * tid; i < H; i += 2 * ntg) {
    const float2 e = d3_row2(OW, OS, OB, c, i, H, p.out_bits, 64);
    const float2 zz = *(const device float2*)(zr + i), w0 = *(const device float2*)(we + i),
                 w1 = *(const device float2*)(we + H + i);
    acc += float4(dot(zz, e), dot(w0, e), dot(w1, e), dot(e, e));
  }
  acc = simd_sum(acc);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float4 s = float4(0.0f);
  for (uint j = 0; j < nsg; ++j) s += red[j];
  cd[f * kD3Cand + k] = s;
}

// d3_cand's dot products in one threadgroup per row (d3_score, then d3_soft's epilogue in the same threadgroup):
// simdgroup k = candidate k, each lane 8 consecutive inputs per 256 (one 32-bit word of 4-bit codes), so a row's 32
// LM-head rows are gathered with z and w_e read once per threadgroup. -> cd[f][k] as d3_cand.
NI_INLINE float4 d3_dots8(const device uint32_t* W, const device bfloat* S, const device bfloat* B, uint c, uint H,
                          uint bits, const device float* zr, const device float* we, uint lane) {
  float4 acc = float4(0.0f);
  const uint gpr = H / 64;
  for (uint k0 = 8 * lane; k0 < H; k0 += 256) {
    const float sc = float(S[size_t(c) * gpr + k0 / 64]), bi = float(B[size_t(c) * gpr + k0 / 64]);
    float e[8];
    if (bits == 4) {
      const uint w = W[size_t(c) * (H / 8) + k0 / 8];
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 8; ++i) e[i] = sc * float((w >> (4 * i)) & 15u) + bi;
    } else {
      const uint2 w = *(const device uint2*)(W + size_t(c) * (H / 4) + k0 / 4);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 8; ++i) e[i] = sc * float(((i < 4 ? w.x : w.y) >> (8 * (i % 4))) & 255u) + bi;
    }
    const float4 e0 = float4(e[0], e[1], e[2], e[3]), e1 = float4(e[4], e[5], e[6], e[7]);
    const float4 z0 = *(const device float4*)(zr + k0), z1 = *(const device float4*)(zr + k0 + 4);
    const float4 a0 = *(const device float4*)(we + k0), a1 = *(const device float4*)(we + k0 + 4);
    const float4 b0 = *(const device float4*)(we + H + k0), b1 = *(const device float4*)(we + H + k0 + 4);
    acc += float4(dot(z0, e0) + dot(z1, e1), dot(a0, e0) + dot(a1, e1), dot(b0, e0) + dot(b1, e1),
                  dot(e0, e0) + dot(e1, e1));
  }
  return acc;
}
// The front rows' children: S = logit_scale[dep] U + <z, E_out[c]>, q = softmax(S / qtemp), the acceptance head's row
// acc_row -> ch_*, expanded = 1. walk = 1 (the chain builder): row 0's node then takes its child (greedy: first
// argmax of S; sampled: the first k whose running sum of q exceeds the walk's uniform for position P + 1 + dep, as
// dfl_walk; qt = 0 at T > 0: the argmax with a one-hot q), writes tokens[P + 1 + dep] and the walk's outputs for
// dfl_accept32 / dfl_depth32 (q, scores [7][32], k [7]; the candidate ids are the cycle's top-32), appends the child
// and makes it the front (none once it is at depth maxdep).
struct D3SoftParams {
  uint H, d, acc_row, has_acc;
  float qtemp, wq, wb, temp;
  uint seed, qt, maxdep, walk;
  uint out_bits;
};
// one row's epilogue on one simdgroup (lane = candidate): c4 = this lane's dot products (cd), ay = <w_y[acc_row], y>
NI_INLINE void d3_children(device D3Tree* t, uint f, float4 c4, float ay, const device int* cand_i,
                           const device float* cand_v, const device float* lscale, device int* tokens,
                           device float* qprob, device int* walk_k, device float* walk_s, constant D3SoftParams& p,
                           uint lane) {
  const uint node = t->front[f];
  const uint dep = uint(t->depth[node]);
  const int c = cand_i[dep * kD3Cand + lane];
  const float S = lscale[dep] * cand_v[dep * kD3Cand + lane] + c4.x;
  const float mx = simd_max(S);
  const float e1 = metal::precise::exp(S - mx);
  const float z1 = simd_sum(e1);
  const float lsm = S - mx - metal::precise::log(z1);
  float q = e1 / z1;
  if (p.qtemp != 1.0f) {
    const float eq = metal::precise::exp((S - mx) / p.qtemp);
    q = eq / simd_sum(eq);
  }
  float a = 1.0f;
  if (p.has_acc) {
    const float inv = metal::precise::rsqrt(c4.w / float(p.H) + 1e-6f);
    const float x = ay + inv * (p.acc_row ? c4.z : c4.y) + p.wq * lsm + p.wb;
    a = 1.0f / (1.0f + metal::precise::exp(-x));
  }
  const uint o = node * kD3Cand + lane;
  t->ch_tok[o] = c;
  t->ch_s[o] = S;
  t->ch_q[o] = q;
  t->ch_a[o] = a;
  if (lane == 0) t->expanded[node] = 1;
  if (!p.walk || f != 0) return;
  // ---- the chain walk ----
  const uint P = t->P;
  uint kb = simd_min(S == mx ? lane : 32u);  // first argmax
  const bool sampled = p.temp > 0.0f;
  if (sampled && p.qt) {
    const float u = dfl_uniform(p.seed, P + 1 + dep, kDflSaltWalk);
    float cum = 0.0f;
    bool found = false;
    kb = 0;
    for (ushort j = 0; j < kD3Cand; ++j) {
      const float qj = simd_shuffle(q, j);
      cum += qj;
      if (!found && qj > 0.0f) kb = j;  // fallback (u above the rounded total): last k with q > 0
      if (!found && cum > u && qj > 0.0f) {
        kb = j;
        found = true;
      }
    }
  }
  qprob[dep * kD3Cand + lane] = sampled ? (p.qt ? q : (lane == kb ? 1.0f : 0.0f)) : 0.0f;
  walk_s[dep * kD3Cand + lane] = S;
  const int ck = simd_shuffle(c, ushort(kb));
  const float lk = simd_shuffle(lsm, ushort(kb));
  const float ak = simd_shuffle(a, ushort(kb));
  if (lane == 0) {
    walk_k[dep] = int(kb);
    tokens[P + 1 + dep] = ck;
    const uint id = t->n_nodes;
    t->parent[id] = int(node);
    t->token[id] = ck;
    t->depth[id] = int(dep + 1);
    t->cand_k[id] = int(kb);
    t->logp[id] = t->logp[node] + lk;
    t->acc[id] = ak;
    t->expanded[id] = 0;
    t->row[id] = int(dep + 1);
    t->n_nodes = id + 1;
    t->front[0] = id;
    t->n_front = dep + 1 < p.maxdep ? 1u : 0u;
  }
}
NI_INLINE float d3_wy_dot(const device float* wy, const device float* y, constant D3SoftParams& p, uint f, uint lane) {
  float a = 0.0f;
  const device float* w = wy + p.acc_row * p.d;
  for (uint i = 4 * lane; i < p.d; i += 128)
    a += dot(*(const device float4*)(w + i), *(const device float4*)(y + size_t(f) * p.d + i));
  return simd_sum(a);
}
// after d3_cand: grid 8 x 64 (simdgroup 1: <w_y, y>; simdgroup 0: the epilogue)
[[kernel]] void d3_soft(device D3Tree* t [[buffer(0)]], const device float* y [[buffer(1)]],
                        const device int* cand_i [[buffer(2)]], const device float* cand_v [[buffer(3)]],
                        const device float4* cd [[buffer(4)]], const device float* lscale [[buffer(5)]],
                        const device float* wy [[buffer(6)]], device int* tokens [[buffer(7)]],
                        device float* qprob [[buffer(8)]], device int* walk_k [[buffer(9)]],
                        device float* walk_s [[buffer(10)]], constant D3SoftParams& p [[buffer(11)]],
                        uint f [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float ay[1];
  if (f >= t->n_front) return;
  if (sg == 1) {
    const float a = d3_wy_dot(wy, y, p, f, lane);
    if (lane == 0) ay[0] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  d3_children(t, f, cd[f * kD3Cand + lane], ay[0], cand_i, cand_v, lscale, tokens, qprob, walk_k, walk_s, p, lane);
}
// d3_cand + d3_soft in one threadgroup per row: simdgroup k gathers candidate k's LM-head row (d3_dots8), simdgroup 0
// then runs the epilogue. grid 8 x 1024.
[[kernel]] void d3_score(device D3Tree* t [[buffer(0)]], const device float* z [[buffer(1)]],
                         const device float* y [[buffer(2)]], const device int* cand_i [[buffer(3)]],
                         const device float* cand_v [[buffer(4)]], const device uint32_t* OW [[buffer(5)]],
                         const device bfloat* OS [[buffer(6)]], const device bfloat* OB [[buffer(7)]],
                         const device float* we [[buffer(8)]], const device float* lscale [[buffer(9)]],
                         const device float* wy [[buffer(10)]], device int* tokens [[buffer(11)]],
                         device float* qprob [[buffer(12)]], device int* walk_k [[buffer(13)]],
                         device float* walk_s [[buffer(14)]], constant D3SoftParams& p [[buffer(15)]],
                         uint f [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 cdl[kD3Cand];
  threadgroup float ay[1];
  if (f >= t->n_front) return;
  const uint node = t->front[f];
  const uint c = uint(cand_i[uint(t->depth[node]) * kD3Cand + sg]);
  const float4 acc = simd_sum(d3_dots8(OW, OS, OB, c, p.H, p.out_bits, z + size_t(f) * p.H, we, lane));
  if (lane == 0) cdl[sg] = acc;
  if (sg == 1) {
    const float a = d3_wy_dot(wy, y, p, f, lane);
    if (lane == 0) ay[0] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  d3_children(t, f, cdl[lane], ay[0], cand_i, cand_v, lscale, tokens, qprob, walk_k, walk_s, p, lane);
}

// The speculative chain walk's step after a decoder pass over a speculated chain (the front: one node per depth from
// walk_at's down to maxdep - 1, each the child of the one above). From walk_at (a node of the real chain, expanded in
// this pass) it takes children as the sequential walk would (greedy: first argmax of S; sampled: the draw of the
// uniform for position P + 1 + j from q, as d3_soft's walk), writing tokens[P + 1 + j] and the walk's outputs of row j,
// and moves down while the chosen child is the front's node at the next depth (its row was computed on the real
// branch). At the first difference it appends the real child y and, unless this is the last pass, makes the next
// pass's front y and below it a new guess (each the choice among the children of this pass's node at the depth
// above, on its stale branch), and returns. In the last pass (last = 1) it finishes the chain on those stale rows
// instead: each remaining position draws from the q of this pass's node at its depth (a valid proposal: the
// acceptance uses the q drawn from, so outputs stay exact; only the draft quality differs from the sequential walk).
// Writes the next pass's indirect dispatch arguments: full (nslot x 3 uints) while the chain is open, else zero
// threadgroups. With maxdep - 1 passes no row is ever stale: the result is the sequential walk's, bitwise.
// grid 1 x 32 (lane = candidate).
struct D3JacParams {
  float temp;
  uint seed, qt, maxdep, last, nslot;
};
// The walk's choice among the children of node src for the position after depth j (lane = candidate): greedy the
// first argmax of S, sampled the first k whose running sum of q exceeds the position's uniform (as d3_soft's walk).
NI_INLINE uint d3_choose(device const D3Tree* t, uint src, uint j, uint P, constant D3JacParams& p, uint lane,
                         thread float& S, thread float& q) {
  S = t->ch_s[src * kD3Cand + lane];
  q = t->ch_q[src * kD3Cand + lane];
  const float mx = simd_max(S);
  uint kb = simd_min(S == mx ? lane : 32u);
  if (p.temp > 0.0f && p.qt) {
    const float u = dfl_uniform(p.seed, P + 1 + j, kDflSaltWalk);
    float cum = 0.0f;
    bool found = false;
    kb = 0;
    for (ushort i = 0; i < kD3Cand; ++i) {
      const float qi = simd_shuffle(q, i);
      cum += qi;
      if (!found && qi > 0.0f) kb = i;  // fallback (u above the rounded total): last k with q > 0
      if (!found && cum > u && qi > 0.0f) {
        kb = i;
        found = true;
      }
    }
  }
  return kb;
}
[[kernel]] void d3_jac(device D3Tree* t [[buffer(0)]], const device int* cand_i [[buffer(1)]],
                       device int* tokens [[buffer(2)]], device float* qprob [[buffer(3)]],
                       device int* walk_k [[buffer(4)]], device float* walk_s [[buffer(5)]],
                       const device uint* full [[buffer(6)]], device uint* next [[buffer(7)]],
                       constant D3JacParams& p [[buffer(8)]], uint lane [[thread_index_in_simdgroup]]) {
  const bool sampled = p.temp > 0.0f;
  const uint P = t->P;
  bool open = t->walk_done == 0;
  if (open) {
    if (lane == 0) t->walk_passes += 1;
    int at[kD3MaxDepth];  // this pass's front node at each depth (-1: none)
    for (uint j = 0; j < kD3MaxDepth; ++j) at[j] = -1;
    for (uint s2 = 0; s2 < t->n_front; ++s2) at[t->depth[t->front[s2]]] = int(t->front[s2]);
    // the walk's position is tracked in registers: nodes lane 0 appends here are not re-read by the other lanes
    uint x = t->walk_at, j = uint(t->depth[x]), nodes = t->n_nodes;
    bool x_exp = t->expanded[x] != 0;
    float x_logp = t->logp[x];
    for (;;) {
      const uint src = x_exp ? x : uint(at[j]);  // x itself, or (last pass) this pass's stale row at its depth
      float S, q;
      const uint kb = d3_choose(t, src, j, P, p, lane, S, q);
      const float mx = simd_max(S);
      const float lsm = S - mx - metal::precise::log(simd_sum(metal::precise::exp(S - mx)));
      qprob[j * kD3Cand + lane] = sampled ? (p.qt ? q : (lane == kb ? 1.0f : 0.0f)) : 0.0f;
      walk_s[j * kD3Cand + lane] = S;
      const int c = simd_shuffle(t->ch_tok[src * kD3Cand + lane], ushort(kb));
      const float lk = simd_shuffle(lsm, ushort(kb)), ak = simd_shuffle(t->ch_a[src * kD3Cand + lane], ushort(kb));
      if (lane == 0) {
        walk_k[j] = int(kb);
        tokens[P + 1 + j] = c;
      }
      if (j + 1 >= p.maxdep) {
        open = false;
        break;
      }
      int y = at[j + 1];
      if (y >= 0 && t->parent[y] == int(x) && t->token[y] == c) {  // the real child is this pass's node: exact row
        if (lane == 0) {  // a guessed node now on the chain: its probabilities
          t->cand_k[y] = int(kb);
          t->logp[y] = x_logp + lk;
          t->acc[y] = ak;
        }
        x = uint(y);
        x_exp = true;
        x_logp += lk;
        ++j;
        continue;
      }
      y = int(nodes++);  // the real child, not scored on its branch yet
      if (lane == 0) {
        t->parent[y] = int(x);
        t->token[y] = c;
        t->depth[y] = int(j + 1);
        t->cand_k[y] = int(kb);
        t->logp[y] = x_logp + lk;
        t->acc[y] = ak;
        t->expanded[y] = 0;
        t->row[y] = int(j + 1);
      }
      if (!p.last) {  // next pass: y and a new guess below it
        uint par = uint(y), nf = 1;
        if (lane == 0) t->front[0] = uint(y);
        for (uint dd = j + 2; dd < p.maxdep; ++dd) {
          float S2, q2;
          const uint k2 = at[dd - 1] >= 0 ? d3_choose(t, uint(at[dd - 1]), dd - 1, P, p, lane, S2, q2) : 0u;
          const int c2 = cand_i[(dd - 1) * kD3Cand + k2];
          const uint nn = nodes++;
          if (lane == 0) {
            t->parent[nn] = int(par);
            t->token[nn] = c2;
            t->depth[nn] = int(dd);
            t->cand_k[nn] = int(k2);
            t->logp[nn] = 0.0f;
            t->acc[nn] = 0.0f;
            t->expanded[nn] = 0;
            t->row[nn] = int(dd);
            t->front[nf] = nn;
          }
          par = nn;
          ++nf;
        }
        if (lane == 0) {
          t->n_front = nf;
          t->walk_at = uint(y);
        }
        break;
      }
      x = uint(y);
      x_exp = false;
      x_logp += lk;
      ++j;
    }
    if (lane == 0) t->n_nodes = nodes;
    if (!open && lane == 0) {
      t->walk_done = 1;
      t->n_front = 0;
    }
  }
  for (uint i = lane; i < p.nslot * 3; i += 32) next[i] = open ? full[i] : (i % 3 == 0 ? 0u : 1u);
}

// ---- top-32 of the draft logits, (value desc, id asc) as dfl_top16 ----
// Lane q < 32 returns the q-th largest of the simdgroup's 256 (every lane holds a sorted run of 8).
NI_INLINE void d3_simd_top32(thread float* v, thread uint* id, uint lane, thread float& ov, thread uint& oi) {
  ov = -INFINITY;
  oi = 0xffffffffu;
  for (uint q = 0; q < 32; ++q) {
    const float mv = simd_max(v[0]);
    const uint mi = simd_min(v[0] == mv ? id[0] : 0xffffffffu);
    if (lane == q) {
      ov = mv;
      oi = mi;
    }
    if (v[0] == mv && id[0] == mi) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 7; ++j) {
        v[j] = v[j + 1];
        id[j] = id[j + 1];
      }
      v[7] = -INFINITY;
      id[7] = 0xffffffffu;
    }
  }
}
// grid (nchunks, rows) x 256 over chunks of 2048: out[(r * nchunks + chunk) * 32 + q], ordered. has_id as dfl_top16
// (0: the index, 1: src_id[r * ld + k], 2: src_id[k]).
[[kernel]] void d3_top32(const device float* src [[buffer(0)]], const device uint* src_id [[buffer(1)]],
                         device float* ov [[buffer(2)]], device uint* oi [[buffer(3)]],
                         constant DflTopParams& p [[buffer(4)]], constant uint& has_id [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[8 * 32];
  threadgroup uint si[8 * 32];
  const uint r = tg.y, base = tg.x * 2048 + tid * 8;
  float v[8];
  uint id[8];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; ++j) {
    const uint k = base + j;
    const bool ok = k < p.n;
    v[j] = ok ? src[size_t(r) * p.ld + k] : -INFINITY;
    id[j] = ok ? (has_id == 2 ? src_id[k] : has_id ? src_id[size_t(r) * p.ld + k] : k) : 0xffffffffu;
  }
  dfl_sort8(v, id);
  float bv;
  uint bi;
  d3_simd_top32(v, id, lane, bv, bi);
  sv[sg * 32 + lane] = bv;
  si[sg * 32 + lane] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 8; ++j) {
    v[j] = sv[lane * 8 + j];
    id[j] = si[lane * 8 + j];
  }
  dfl_sort8(v, id);
  d3_simd_top32(v, id, lane, bv, bi);
  const size_t o = (size_t(r) * p.nchunks + tg.x) * 32 + lane;
  ov[o] = bv;
  oi[o] = bi;
}
