#include "common.h"

// ---- 8-row verify: fused small passes (fewer dependent dispatches per layer) ----
// Each kernel here does the work of two or more verify8.metal passes in one dispatch, with the same arithmetic in
// the same order (bitwise the same outputs): a dependent dispatch costs its launch plus a pipeline drain on top of
// its own latency, and the verify chains ~600 of them.

// MoE combine + the next X8 table: x[t] += sum_j w[t][j] out[t k + j] + w[t][k] sh_out[t] (moe8_combine's order),
// then x8_prep2's RMSNorm table of the new rows (its sum-of-squares order, its group-sum order). One threadgroup per
// token row (the norm needs the whole row): the combined row is staged in threadgroup memory, simdgroup 0 forms the
// row's norm, every thread then writes 8 consecutive inputs into the table (2-byte stores: a row is every other half
// of the table's 16-byte lane runs). Rows t >= R get zeros. grid 8 x N / 4 threads (N <= 2048).
struct Moe8CombineX8Params {
  uint N, k;     // hidden, routed experts per row
  uint R;        // live rows
  float eps;
  uint unroll;   // x8_prep2's unroll (V8Knobs::prepu): 4 independent sum-of-squares chains
};
[[kernel]] void moe8_combine_x8(const device float4* out [[buffer(0)]], const device float4* sh_out [[buffer(1)]],
                                const device float* wts [[buffer(2)]], device float4* x [[buffer(3)]],
                                const device bfloat* w [[buffer(4)]], device half* table [[buffer(5)]],
                                device float* sums [[buffer(6)]], constant Moe8CombineX8Params& p [[buffer(7)]],
                                uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 xs[512];
  threadgroup float inv_s;
  const uint n4 = p.N / 4, K = p.N;
  const bool live = t < p.R;
  if (live) {
    // moe8_combine's code as it is (one float4 per thread): written otherwise, the compiler orders the sum otherwise
    const uint i = tid;
    if (i < n4) {
      float4 acc = float4(0.0f);
      for (uint j = 0; j < p.k; ++j) acc += wts[t * (p.k + 1) + j] * out[(t * p.k + j) * n4 + i];
      acc += wts[t * (p.k + 1) + p.k] * sh_out[t * n4 + i];
      x[t * n4 + i] += acc;
      xs[i] = x[t * n4 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float ss = 0.0f;
      if (p.unroll && K % 512 == 0) {
        float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
        for (uint i = lane; i < K / 4; i += 128) {
          const float4 v0 = xs[i], v1 = xs[i + 32], v2 = xs[i + 64], v3 = xs[i + 96];
          a0 = fma(v0, v0, a0);
          a1 = fma(v1, v1, a1);
          a2 = fma(v2, v2, a2);
          a3 = fma(v3, v3, a3);
        }
        const float4 a = (a0 + a1) + (a2 + a3);
        ss = (a.x + a.y) + (a.z + a.w);
      } else {
        for (uint i = lane; i < K / 4; i += 32) {
          const float4 v = xs[i];
          ss += dot(v, v);
        }
      }
      ss = simd_sum(ss);
      if (lane == 0) inv_s = metal::precise::rsqrt(ss / float(K) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // thread q < K / 8: inputs 8q .. 8q + 7; a group's 8 octets are 8 consecutive lanes of one simdgroup, added in
  // x8_prep2's order. A thread's 8 halfs are summed in sequence with an identity shuffle between the additions: fast
  // math would otherwise add them as two 4-vectors in some contexts (a different rounding).
  const float inv = live ? inv_s : 0.0f;
  const threadgroup float* xf = (const threadgroup float*)xs;
  const uint q = tid, k0 = q * 8;
  const bool on = q < K / 8;
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  if (live && on)
    for (int i = 0; i < 8; ++i) v[i] = xf[k0 + i] * inv * float(w[k0 + i]);
  half hv[8];
  for (int i = 0; i < 8; ++i) hv[i] = half(v[i]);
  float sh = float(hv[0]);
  for (int i = 1; i < 8; ++i) sh = simd_shuffle(sh, ushort(lane)) + float(hv[i]);
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  if (on) {
    const uint g = k0 / 64;
    if ((lane & 7) == 0) sums[g * kV8 + t] = sh;
    for (int i = 0; i < 8; ++i) table[v8_tab(g, (k0 % 64) + i, t)] = hv[i];
  }
}

// MoE router top-k + plan (moe_topk256_rows + moe8_plan, their shared bodies): simdgroup t routes row t, the plan
// reads the rows' ids from threadgroup memory. E = 256, one threadgroup of 256 threads (R <= 8 simdgroups route).
[[kernel]] void moe8_topk_plan(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                               device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                               constant TopkParams& tp [[buffer(4)]], device uint* tiles [[buffer(5)]],
                               device uint* ntiles [[buffer(6)]], device uint* slot [[buffer(7)]],
                               device uint* args [[buffer(8)]], constant Moe8PlanParams& p [[buffer(9)]],
                               device uint* stats [[buffer(10)]], device uint* sm [[buffer(11)]],
                               const device uint* rows_lim [[buffer(12)]],
                               uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ids[64];
  threadgroup uint cnt[2][32];
  threadgroup uint total[2];
  if (sg < p.R) moe_topk256_row(logits, shared, idx, wts, tp, sg, lane, ids);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint R = p.rlim ? min(p.R, rows_lim[0]) : p.R;
  moe8_plan_body(ids, R, tiles, ntiles, slot, args, p, stats, sm, tid, ntg, sg, lane, cnt, total);
}
