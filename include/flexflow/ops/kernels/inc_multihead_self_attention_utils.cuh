#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H

#include "flexflow/inference.h"

namespace FlexFlow {

////////////////basic datatype//////////////////////
struct half4 {
  half x;
  half y;
  half z;
  half w;
};

struct half8 {
  half x;
  half y;
  half z;
  half w;
  half a;
  half b;
  half c;
  half d;
};
struct float8 {
  float x;
  float y;
  float z;
  float w;
  float a;
  float b;
  float c;
  float d;
};

////////////////data type///////////////
template <typename DT, int VECPSIZE>
struct VEC_K {};
template <>
struct VEC_K<float, 1> {
  using Type = float;
};
template <>
struct VEC_K<float, 2> {
  using Type = float2;
};
template <>
struct VEC_K<float, 4> {
  using Type = float4;
};
template <>
struct VEC_K<half, 1> {
  using Type = half;
};
template <>
struct VEC_K<half, 2> {
  using Type = half2;
};
template <>
struct VEC_K<half, 4> {
  using Type = half4;
};

// data type for QK production
template <typename T>
struct Vec_fp32_ {};

template <>
struct Vec_fp32_<float> {
  using Type = float;
};
template <>
struct Vec_fp32_<float2> {
  using Type = float2;
};
template <>
struct Vec_fp32_<float4> {
  using Type = float4;
};
template <>
struct Vec_fp32_<half> {
  using Type = float;
};
template <>
struct Vec_fp32_<half2> {
  using Type = float2;
};
template <>
struct Vec_fp32_<half4> {
  using Type = float4;
};
template <>
struct Vec_fp32_<half8> {
  using Type = float8;
};

template <typename DT>
struct VEC_V {};
template <>
struct VEC_V<float> {
  using Type = float4;
};
template <>
struct VEC_V<half> {
  using Type = half8;
};

////////////////data structures half///////////////

////////////////////////////////////floating point
/// operations///////////////////////////////////////////

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
  return Acc{}; // for compile
}
template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

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

// template <>
// inline __device__ float4 mul(half4 a, half4 b) {
//   float4 c;
//   c.x = a.x * b.x;
//   c.y = a.y * b.y;
//   c.z = a.z * b.z;
//   c.w = a.w * b.w;
//   return c;
// }

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

inline __device__ float8 fma(float a, float8 f1, float8 f2) {
  float8 res;
  res.x = fma(a, f1.x, f2.x);
  res.y = fma(a, f1.y, f2.y);
  res.z = fma(a, f1.z, f2.z);
  res.w = fma(a, f1.w, f2.w);
  res.a = fma(a, f1.a, f2.a);
  res.b = fma(a, f1.b, f2.b);
  res.c = fma(a, f1.c, f2.c);
  res.d = fma(a, f1.d, f2.d);
  return res;
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

inline __device__ float8 add(float8 f1, float8 f2) {
  float8 res;
  res.x = add(f1.x, f2.x);
  res.y = add(f1.y, f2.y);
  res.z = add(f1.z, f2.z);
  res.w = add(f1.w, f2.w);
  res.a = add(f1.a, f2.a);
  res.b = add(f1.b, f2.b);
  res.c = add(f1.c, f2.c);
  res.d = add(f1.d, f2.d);
  return res;
}

inline __device__ float sum(float v) {
  return v;
}

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
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

inline __device__ float cast_to_float(half u) {
  return __half2float(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(half2 u) {
  float2 tmp;
  tmp.x = __half2float(u.x);
  tmp.y = __half2float(u.y);
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(half4 u) {
  float4 tmp;
  tmp.x = __half2float(u.x);
  tmp.y = __half2float(u.y);
  tmp.z = __half2float(u.z);
  tmp.w = __half2float(u.w);
  return tmp;
}
inline __device__ float8 cast_to_float(half8 u) {
  float8 tmp;
  tmp.x = __half2float(u.x);
  tmp.y = __half2float(u.y);
  tmp.z = __half2float(u.z);
  tmp.w = __half2float(u.w);
  tmp.a = __half2float(u.a);
  tmp.b = __half2float(u.b);
  tmp.c = __half2float(u.c);
  tmp.d = __half2float(u.d);
  return tmp;
}

inline __device__ void convert_from_float(float4 &dst, float4 src) {
  dst = src;
}
inline __device__ void convert_from_float(float &dst, float src) {
  dst = src;
}
inline __device__ void convert_from_float(float2 &dst, float2 src) {
  dst = src;
}
inline __device__ void convert_from_float(float8 &dst, float8 src) {
  dst = src;
}

inline __device__ void convert_from_float(half4 &dst, float4 src) {
  dst.x = __float2half(src.x);
  dst.y = __float2half(src.y);
  dst.z = __float2half(src.z);
  dst.w = __float2half(src.w);
}

inline __device__ void convert_from_float(half8 &dst, float8 src) {
  dst.x = __float2half(src.x);
  dst.y = __float2half(src.y);
  dst.z = __float2half(src.z);
  dst.w = __float2half(src.w);
  dst.a = __float2half(src.a);
  dst.b = __float2half(src.b);
  dst.c = __float2half(src.c);
  dst.d = __float2half(src.d);
}
inline __device__ void convert_from_float(half2 &dst, float2 src) {
  dst.x = __float2half(src.x);
  dst.y = __float2half(src.y);
}
inline __device__ void convert_from_float(half &dst, float src) {
  dst = __float2half(src);
}

//////////////////////////////////////utils///////////////////////////////////////////////

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
  // use float32 to get better accuracy
  using Vec_sum = typename Vec_fp32_<K_vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  Vec_sum qk_vec =
      mul<Vec_sum, Vec_sum, Vec_sum>(cast_to_float(q[0]), cast_to_float(k[0]));
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = FlexFlow::fma(cast_to_float(q[ii]), cast_to_float(k[ii]), qk_vec);
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

template <typename DT>
inline size_t smem_size_in_bytes(int hidden_size_per_head,
                                 int max_sequence_length,
                                 int threads_per_value,
                                 int threads_per_block) {
  // The amount of shared memory needed to store the Q*K^T values in float.

  size_t qk_sz = div_up(max_sequence_length + 1, 4) * 16;
  size_t logits_sz = qk_sz;

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;
  size_t q_size = hidden_size_per_head * sizeof(DT);

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;
  // The amount of storage needed to finalize the outputs.
  size_t red_sz = rows_per_red * hidden_size_per_head * sizeof(float) / 2;
  // The max.
  return max(softmax_sz, red_sz) + q_size;
}

template <typename DT>
inline void smem_size_in_bytes_tree(int hidden_size_per_head,
                                    int max_sequence_length,
                                    int threads_per_value,
                                    int threads_per_block,
                                    TreeVerifyBatchConfig const *bc,
                                    int shared_mem[]) {

  int max_query_length = 0;
  int max_total_length = 0;
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    max_query_length =
        max(max_query_length, bc->requestsInfo[i].num_tokens_in_batch);
    max_total_length = max(max_total_length,
                           bc->requestsInfo[i].first_token_depth_in_request +
                               bc->requestsInfo[i].num_tokens_in_batch);
  }

  // todo fix this
  int max_qk_length = max_query_length;

  // The amount of shared memory needed to store the Q*K^T values in float.
  size_t qk_sz = div_up(max_qk_length + 1, 4) * 16;

  size_t logits_sz = qk_sz;

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;

  size_t q_size = hidden_size_per_head * sizeof(DT);

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;
  // The amount of storage needed to finalize the outputs.
  // use 4
  size_t red_sz = rows_per_red * hidden_size_per_head * sizeof(float) / 2;
  // The max.
  shared_mem[0] = qk_sz;
  shared_mem[1] = softmax_sz + red_sz + q_size;
}

template <typename T, int Dh>
struct threads_per_value_t {
  static int const value = Dh * sizeof(T) / 16;
};

} // namespace FlexFlow
#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_UTILS_H