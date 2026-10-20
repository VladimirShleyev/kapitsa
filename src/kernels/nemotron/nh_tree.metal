// KFlash trees for the Nemotron-H block drafters (DSpark, LiLiCorr): each drafter's candidate lattice as conditional
// log-probabilities (tlq), a best-first tree over it under the measured row budget, the tree's acceptance after the
// target's verify, and the accepted path's verify features for the drafter's ring commit. The tree buffer's layout
// (kKf*) is in nh_common.h; the verify's tree kernels (nh_ssm1_tree, nh_attn_partial_tree, nh_kv_tree_compact) are in
// nh_ops.metal.

// ---- DSpark's lattice: tlq[s][p][c] = the Markov head's corrected log-probability of candidate c of slot s after
// candidate p of slot s - 1 (slot 0: after the anchor, p = 0), normalized over the slot's K candidates plus the mass
// outside them as nh_dsp_walk normalizes its picks. grid: (1 + (S - 1) K) x (32 K) threads: threadgroup g is slot 0
// for g = 0, else slot 1 + (g - 1) / K after candidate (g - 1) % K; simdgroup c computes candidate c. ----
struct NhDspTableP {
  uint S, K, R, P;
  float gs;
};
[[kernel]] void nh_dsp_table(const device int* cand [[buffer(0)]], const device float* clp [[buffer(1)]],
                             const device ushort* w1 [[buffer(2)]], const device uchar* w2q [[buffer(3)]],
                             const device uchar* w2s [[buffer(4)]], const device int* tokens [[buffer(5)]],
                             device float* tlq [[buffer(6)]], constant NhDspTableP& p [[buffer(7)]],
                             uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float e[1024];
  threadgroup float sc[32];
  const uint s = g == 0 ? 0u : 1u + (g - 1) / p.K, pp = g == 0 ? 0u : (g - 1) % p.K;
  const uint prev = uint(g == 0 ? tokens[p.P] : cand[(s - 1) * p.K + pp]);
  for (uint k = tid; k < p.R; k += ntg) e[k] = as_type<float>(uint(w1[size_t(prev) * p.R + k]) << 16);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint id = uint(cand[s * p.K + c]);
  float d = 0.0f;
  for (uint b = lane; b < p.R / 16; b += 32) {  // one 16-element block per lane: byte j = elements j, j + 8
    const uint2 q2 = *(const device uint2*)(w2q + size_t(id) * (p.R / 2) + b * 8);
    const float bs = nh_ue4m3(w2s[size_t(id) * (p.R / 16) + b]);
    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
      const uint byte = ((j < 4 ? q2.x : q2.y) >> (8 * (j & 3))) & 0xFFu;
      acc += nh_fp4_lut[byte & 15u] * e[b * 16 + j] + nh_fp4_lut[byte >> 4] * e[b * 16 + j + 8];
    }
    d += acc * bs;
  }
  d = simd_sum(d);
  if (lane == 0) sc[c] = clp[s * p.K + c] + d * p.gs;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float mx = sc[0], base = 0.0f, den = 0.0f;
    for (uint k = 1; k < p.K; ++k) mx = max(mx, sc[k]);
    for (uint k = 0; k < p.K; ++k) {
      den += metal::precise::exp(sc[k] - mx);
      base += metal::precise::exp(clp[s * p.K + k]);
    }
    den += max(0.0f, 1.0f - base) * metal::precise::exp(-mx);
    const float lden = mx + metal::precise::log(den);
    for (uint k = 0; k < p.K; ++k) tlq[(s * p.K + pp) * p.K + k] = sc[k] - lden;
  }
}

// ---- LiLiCorr's lattice: tlq[s][p][c] = log softmax over the slot's K candidates of the head's transition scores
// (slot 0: ls in[0, c] . aout against the anchor; slot s: ls in[s, c] . out[s - 1, p]), plus the log of the drafter's
// probability mass on those K candidates (clp: log-probabilities over its full head), so that a slot whose K
// candidates miss the target's token is not counted as certain. grid: as nh_dsp_table. ----
struct NhLiliTableP {
  uint S, K, D;
  float ls;
};
[[kernel]] void nh_lili_table(const device float* in [[buffer(0)]], const device float* outv [[buffer(1)]],
                              const device float* aout [[buffer(2)]], const device float* clp [[buffer(3)]],
                              device float* tlq [[buffer(4)]], constant NhLiliTableP& p [[buffer(5)]],
                              uint g [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint c [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sc[32];
  const uint s = g == 0 ? 0u : 1u + (g - 1) / p.K, pp = g == 0 ? 0u : (g - 1) % p.K;
  const device float* a = in + (size_t(s) * p.K + c) * p.D;
  const device float* b = s == 0 ? aout : outv + (size_t(s - 1) * p.K + pp) * p.D;
  float d = 0.0f;
  for (uint o = lane; o < p.D; o += 32) d += a[o] * b[o];
  d = simd_sum(d);
  if (lane == 0) sc[c] = p.ls * d;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float mx = sc[0], den = 0.0f, mass = 0.0f;
    for (uint k = 1; k < p.K; ++k) mx = max(mx, sc[k]);
    for (uint k = 0; k < p.K; ++k) {
      den += metal::precise::exp(sc[k] - mx);
      mass += metal::precise::exp(clp[s * p.K + k]);
    }
    const float l = mx + metal::precise::log(den) - metal::precise::log(min(1.0f, mass));
    for (uint k = 0; k < p.K; ++k) tlq[(s * p.K + pp) * p.K + k] = sc[k] - l;
  }
}

// ---- the tree builder: best-first over the lattice ----
// A node's score is the sum over its root path of log p, p = sigmoid(alpha_d + beta_d logit(q)) the calibrated
// probability that the target accepts a node of depth d whose drafter probability is q (the KFlash budget fits alpha
// and beta online from the acceptance's records; 0 and 1 give q itself). Up to 7 nodes are taken in decreasing score
// (a child never scores above its parent, so rows are in insertion order); the verify's rows n (the root and the first
// n - 1 nodes) maximize (1 + the nodes' summed path probabilities) / cost[n], the cycle's measured cost with n rows
// (force_n > 0: n = force_n, the cost table's exploration). The nodes' tokens go to tokens[P + r]; the tree's rows,
// parents, depths, masks and log q to the tree buffer, n to live[0]. grid: 1 x 32 threads.
struct NhKfParams {  // written by the host (src/model/kflash_budget.h) between cycles
  float alpha[8], beta[8];  // calibration of depth d = 1 .. 7
  float cost[9];            // cost[n], n = 1 .. 8 verify rows
  int force_n;
};
NI_INLINE float nh_kf_lcal(float lq, float a, float b) {
  const float q = min(metal::precise::exp(lq), 0.999999f);
  const float z = a + b * (lq - metal::precise::log(1.0f - q));
  return -softplus(-z);  // log sigmoid(z)
}
struct NhKfBuildP {
  uint S, K, P;
};
[[kernel]] void nh_kf_build(const device float* tlq [[buffer(0)]], const device int* cand [[buffer(1)]],
                            const device NhKfParams& kp [[buffer(2)]], device int* tree [[buffer(3)]],
                            device int* tokens [[buffer(4)]], device uint* live [[buffer(5)]],
                            constant NhKfBuildP& p [[buffer(6)]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float fs[7 * 32];  // frontier: the node's path score (-inf once taken)
  threadgroup float fq[7 * 32];  // its own log q
  threadgroup uint fe[7 * 32];   // parent row | slot << 4 | candidate << 8
  threadgroup float ns[8];       // the rows' path scores
  threadgroup uint nf, nodes;
  const uint S = p.S, K = p.K;
  for (uint c = lane; c < K; c += 32) {  // the root's children: slot 0 after the anchor
    const float lq = tlq[c];
    fs[c] = nh_kf_lcal(lq, kp.alpha[1], kp.beta[1]);
    fq[c] = lq;
    fe[c] = c << 8;
  }
  if (lane == 0) {
    nf = K;
    nodes = 1;
    ns[0] = 0.0f;
    tree[kKfPar] = -1;
    tree[kKfDep] = 0;
    tree[kKfMask] = 1;
    tree[kKfLq] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 1; r < 8; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = lane; i < nf; i += 32)
      if (fs[i] > bv) {
        bv = fs[i];
        bi = i;
      }
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(bv, off);
      const uint oi = simd_shuffle_xor(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (bv == -INFINITY) break;
    const uint e = fe[bi], pr = e & 15u, sl = (e >> 4) & 15u, c = e >> 8;
    const uint base = nf;
    if (lane == 0) {
      fs[bi] = -INFINITY;
      tree[kKfPar + r] = int(pr);
      tree[kKfDep + r] = int(sl + 1);
      tree[kKfMask + r] = tree[kKfMask + pr] | int(1u << r);
      tree[kKfLq + r] = as_type<int>(fq[bi]);
      tokens[p.P + r] = cand[sl * K + c];
      ns[r] = bv;
      nodes = r + 1;
    }
    const bool push = sl + 1 < S && r + 1 < 8;  // (the last node's children would never be taken)
    if (push)
      for (uint c2 = lane; c2 < K; c2 += 32) {
        const float lq = tlq[((sl + 1) * K + c) * K + c2];
        fs[base + c2] = bv + nh_kf_lcal(lq, kp.alpha[sl + 2], kp.beta[sl + 2]);
        fq[base + c2] = lq;
        fe[base + c2] = r | ((sl + 1) << 4) | (c2 << 8);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && push) nf = base + K;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    const uint m = nodes;
    uint n = 1;
    if (kp.force_n > 0) {
      n = min(uint(kp.force_n), m);
    } else {
      float E = 1.0f, best = E / kp.cost[1];
      for (uint r = 1; r < m; ++r) {
        E += metal::precise::exp(ns[r]);
        const float v = E / kp.cost[r + 1];
        if (v > best) {
          best = v;
          n = r + 1;
        }
      }
    }
    tree[kKfN] = int(n);
    live[0] = n;
    for (uint r = m; r < 8; ++r) {  // rows past the tree: a well-formed filler chain (never live)
      tree[kKfPar + r] = int(r - 1);
      tree[kKfDep + r] = min(tree[kKfDep + r - 1] + 1, int(S));
      tree[kKfMask + r] = tree[kKfMask + r - 1] | int(1u << r);
      tree[kKfLq + r] = 0;
      tokens[p.P + r] = cand[min(r - 1, S - 1) * K];
    }
  }
}

// ---- the tree's acceptance ----
// samples[r] = the target's token after row r, drawn at position P + depth(r) (sample_stage2_dstep), so each is the
// plain sampler's draw after the row's root path. From the root, descend into the child whose token equals the current
// row's sample while there is one: the path, a = its drafts; tokens[P + 1 .. P + a] = the path's tokens and
// tokens[P + 1 + a] = its last row's sample. Greedy, this is plain decoding's output; sampled, the plain sampler's
// position-keyed draws, whatever the tree. Also: the path in the tree buffer and in ppath (the next pass's pending
// rows), acc[0] = a, and the calibration records (every node whose parent is on the path: depth, log q, accepted).
// One thread.
[[kernel]] void nh_accept_tree(const device int* samples [[buffer(0)]], device int* tokens [[buffer(1)]],
                               device int* acc [[buffer(2)]], device int* tree [[buffer(3)]],
                               constant NhAcceptP& p [[buffer(4)]], const device uint* live [[buffer(5)]],
                               device int* ppath [[buffer(6)]]) {
  const uint n = min(uint(tree[kKfN]), min(live[0], 8u));
  int tok[8];
  for (uint r = 0; r < 8; ++r) tok[r] = r < n ? tokens[p.P + r] : -1;
  uint path[9];
  path[0] = 0;
  uint a = 0, cur = 0;
  for (;;) {
    const int sm = samples[cur];
    uint nxt = 0;
    for (uint c = cur + 1; c < n; ++c)
      if (tree[kKfPar + c] == int(cur) && tok[c] == sm) {
        nxt = c;
        break;
      }
    if (nxt == 0) break;
    path[++a] = nxt;
    cur = nxt;
  }
  uint onpath = 0;
  for (uint i = 0; i <= a; ++i) onpath |= 1u << path[i];
  uint nrec = 0;
  for (uint c = 1; c < n; ++c) {
    const uint par = uint(tree[kKfPar + c]);
    if (!((onpath >> par) & 1u)) continue;
    tree[kKfRecDep + nrec] = tree[kKfDep + c];
    tree[kKfRecLq + nrec] = tree[kKfLq + c];
    tree[kKfRecHit + nrec] = tok[c] == samples[par] ? 1 : 0;
    ++nrec;
  }
  tree[kKfNrec] = int(nrec);
  for (uint i = 1; i <= a; ++i) tokens[p.P + i] = tok[path[i]];
  tokens[p.P + 1 + a] = samples[path[a]];
  acc[0] = int(a);
  tree[kKfA] = int(a);
  for (uint i = 0; i <= a; ++i) {
    tree[kKfPath + i] = int(path[i]);
    ppath[i] = int(path[i]);
  }
}

// ---- the accepted path's verify features (rows path[0 .. m) of feat) as rows 0 .. m of out, for the drafter's ring
// commit (the rows plain decoding of those tokens would have captured). grid: (m, ceil(ld / 256)) x 256 ----
struct NhKfGatherP {
  uint ld;
};
[[kernel]] void nh_kf_gather(const device float* feat [[buffer(0)]], device float* out [[buffer(1)]],
                             const device int* path [[buffer(2)]], constant NhKfGatherP& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]]) {
  const uint i = tg.x, c = tg.y * 256 + t2.x;
  if (c < p.ld) out[size_t(i) * p.ld + c] = feat[size_t(path[i]) * p.ld + c];
}
