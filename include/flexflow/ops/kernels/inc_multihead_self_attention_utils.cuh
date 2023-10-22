#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H

namespace FlexFlow {

template <typename T, int Dh>
struct Qk_vec_k_ {};

template <>
struct Qk_vec_k_<float, 32> {
  using Type = float;
};
template <>
struct Qk_vec_k_<float, 64> {
  using Type = float2;
};
template <>
struct Qk_vec_k_<float, 128> {
  using Type = float4;
};
template <>
struct Qk_vec_k_<float, 256> {
  using Type = float4;
};

template <>
struct Qk_vec_k_<half, 32> {
  using Type = half;
};
template <>
struct Qk_vec_k_<half, 64> {
  using Type = half;
};
template <>
struct Qk_vec_k_<half, 128> {
  using Type = half;
};
template <>
struct Qk_vec_k_<half, 256> {
  using Type = half;
};

template <typename T, int THREADS_PER_KEY>
struct K_vec_k_ {};

template <>
struct K_vec_k_<float, 4> {
  using Type = float4;
};
template <>
struct K_vec_k_<float, 2> {
  using Type = float2;
};
template <>
struct K_vec_k_<float, 1> {
  using Type = float4;
};

template <>
struct K_vec_k_<half, 4> {
  using Type = float;
};
template <>
struct K_vec_k_<half, 2> {
  using Type = float2;
};
template <>
struct K_vec_k_<half, 1> {
  using Type = float4;
};

template <typename T>
struct K_vec_acum_fp32_ {};

template <>
struct K_vec_acum_fp32_<float> {
  using Type = float;
};
template <>
struct K_vec_acum_fp32_<float2> {
  using Type = float2;
};
template <>
struct K_vec_acum_fp32_<float4> {
  using Type = float4;
};

template <>
struct K_vec_acum_fp32_<half> {
  using Type = half;
};

template <typename T, int V_VEC_SIZE>
struct V_vec_k_ {};

template <>
struct V_vec_k_<float, 1> {
  using Type = float;
};
template <>
struct V_vec_k_<float, 2> {
  using Type = float2;
};
template <>
struct V_vec_k_<float, 4> {
  using Type = float4;
};

template <>
struct V_vec_k_<half, 1> {
  using Type = half;
};
template <>
struct V_vec_k_<half, 2> {
  using Type = half;
};
template <>
struct V_vec_k_<half, 4> {
  using Type = half;
};

template <typename T>
struct V_vec_acum_fp32_ {};

template <>
struct V_vec_acum_fp32_<float> {
  using Type = float;
};
template <>
struct V_vec_acum_fp32_<float2> {
  using Type = float2;
};
template <>
struct V_vec_acum_fp32_<float4> {
  using Type = float4;
};

template <>
struct V_vec_acum_fp32_<half> {
  using Type = float;
};

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
  return Acc{}; // for compile
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(float a, float b, float c) {
  return a * b + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ float4 fma(half a, float4 b, float4 c) {
  assert(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, float b) {
  return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float sum(float v) {
  return v;
}

template <typename T>
inline __device__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

inline __device__ float cast_to_float(float u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u) {
  return u;
}

template <typename T>
inline __device__ void zero(T &dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;
#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(K_vec const (&q)[N], K_vec const (&k)[N]) {
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
template <typename T, int THREADS_PER_KEY>
struct Qk_dot {
  template <typename K_vec, int N>
  static inline __device__ float dot(K_vec const (&q)[N], K_vec const (&k)[N]) {
    return qk_dot_<THREADS_PER_KEY>(q, k);
  }
};

template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float *red_smem, float sum) {

  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

// Parallel reduction inside the warp.
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

} // namespace FlexFlow
#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H