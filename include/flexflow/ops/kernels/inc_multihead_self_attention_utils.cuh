#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H

namespace FlexFlow {

template <typename T, int THREADS_PER_KEY>
struct Qk_dot {
  template <typename K_vec, int N>
  static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N]) {
    return qk_dot_<THREADS_PER_KEY>(q, k);
  }
};

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N]) {
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
  using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
  using K_vec_acum = K_vec;
#endif
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}
} // namespace FlexFlow
#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H