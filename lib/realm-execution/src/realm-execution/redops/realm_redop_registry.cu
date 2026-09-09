#include "realm-execution/redops/realm_redop_registry.h"
#include "realm-execution/redops/redop_id_t.h"

namespace FlexFlow {

// Reduction operators and related infrastructure borrowed from Legion. We
// maintain the Legion naming scheme to maximizing compatibility with the
// existing code, despite not otherwise relying or using Legion in any way.
// https://gitlab.com/StanfordLegion/legion/-/blob/5263aeff477fb94239c50d9306d58c4244e9fc38/runtime/legion/api/redop.inl#L31
#if !defined(__cpp_lib_atomic_ref) || (__cpp_lib_atomic_ref < 201806L)
// We only need this crap if we're using a version of c++ < 20
// Starting with c++20 we can do all this the right way with atomic_ref
namespace TypePunning {
// The tenth circle of hell is reserved for members of the C++ committee
// that decided to deviate from C's support for type punning unions.
// Add on to it the fact that it took them 9 fucking years to realize
// that they needed std::atomic_ref and it's plain to see they are all
// just a bunch of idiots that should never be allowed near a programming
// language standard ever again. They've clearly never written lock-free
// code in their lives.
template <typename T>
class Pointer {
public:
  Pointer(void *p) : pointer(convert(p)) {}
  static inline T *convert(void *p) {
    T *ptr = nullptr;
    static_assert(sizeof(ptr) == sizeof(p));
    memcpy(&ptr, &p, sizeof(p));
    return ptr;
  }
  inline operator T *(void) const {
    return (T *)pointer;
  }
  inline T operator*(void) const {
    return *pointer;
  }
  inline T operator[](size_t off) const {
    return pointer[off];
  }

private:
  T volatile *const pointer;
};
template <typename T, size_t ALIGNMENT = alignof(T)>
class AlignedPointer {
public:
  AlignedPointer(void *p) : off(align(p)), pointer(convert(p, off)) {}
  static inline T *convert(void *p, size_t off) {
    uint8_t *p1 = nullptr;
    static_assert(sizeof(p1) == sizeof(p));
    memcpy(&p1, &p, sizeof(p));
    p1 = p1 - off;
    T *p2 = nullptr;
    static_assert(sizeof(p1) == sizeof(p2));
    memcpy(&p2, &p1, sizeof(p1));
    return p2;
  }
  static inline size_t align(void *p) {
    uintptr_t ptr;
    static_assert(sizeof(ptr) == sizeof(p));
    memcpy(&ptr, &p, sizeof(ptr));
    return ptr % ALIGNMENT;
  }
  inline operator T *(void) const {
    return (T *)pointer;
  }
  inline T operator*(void) const {
    return *pointer;
  }
  inline size_t offset(void) const {
    return off;
  }

private:
  size_t off;
  T volatile *const pointer;
};
template <typename T1, typename T2>
class Alias {
public:
  inline void load(Pointer<T1> const &pointer, size_t off = 0) {
    T1 value = pointer[off];
    memcpy(buffer, (void *)&value, sizeof(T1));
  }
  template <size_t ALIGNMENT>
  inline void load(AlignedPointer<T1, ALIGNMENT> const &pointer) {
    T1 value = *pointer;
    memcpy(buffer, (void *)&value, sizeof(T1));
  }
  inline T1 as_one(void) const {
    T1 result;
    memcpy((void *)&result, buffer, sizeof(result));
    return result;
  }
  inline T2 as_two(void) const {
    T2 result;
    memcpy((void *)&result, buffer, sizeof(result));
    return result;
  }
  inline Alias &operator=(T2 rhs) {
    memcpy(buffer, (void *)&rhs, sizeof(rhs));
    return *this;
  }

private:
  // Make this one private so it is can never be called
  inline Alias &operator=(T1 rhs) {
    memcpy(buffer, (void *)&rhs, sizeof(rhs));
    return *this;
  }
  static_assert(sizeof(T1) == sizeof(T2));
  uint8_t buffer[sizeof(T1)];
};
}; // namespace TypePunning
#endif

// Define a prefix for annotating functions for CUDA compilation
#if defined(__CUDACC__) || defined(__HIPCC__)
#define __LEGION_CUDA_HD__ __host__ __device__
#else
#define __LEGION_CUDA_HD__
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
    // We have these functions here because calling memcpy (per the
// insistence of the idiots on the C++ standards committee) on the
// GPU is a terrible idea since it will spill the data out of registers
// and into local memory in order to do the memcpy. Hence we tell
// the compiler exactly what we mean using bit operations and inline PTX.

__device__ __forceinline__ bool __uint2bool(unsigned int value,
                                            unsigned offset) {
  value = value >> (8 * offset);
  return ((value & 0xFF) != 0);
}

__device__ __forceinline__ unsigned int
    __bool2uint(unsigned int previous, bool value, unsigned offset) {
  unsigned int next = value;
  next = next << (8 * offset);
  unsigned int mask = 0xFF;
  mask = mask << (8 * offset);
  previous = previous & (~mask);
  return previous | next;
}

__device__ __forceinline__ uint8_t __uint2ubyte(unsigned int value,
                                                unsigned offset) {
  value = value >> (8 * offset);
  return uint8_t(value & 0xFF);
}

__device__ __forceinline__ unsigned int
    __ubyte2uint(unsigned int previous, uint8_t value, unsigned offset) {
  unsigned int next = value;
  next = next << (8 * offset);
  unsigned int mask = 0xFF;
  mask = mask << (8 * offset);
  previous = previous & (~mask);
  return previous | next;
}

__device__ __forceinline__ int8_t __int2byte(int value, unsigned offset) {
  value = value >> (8 * offset);
  return int8_t(value & 0xFF);
}

__device__ __forceinline__ int
    __byte2int(int previous, int8_t value, unsigned offset) {
  int next = value;
  next = next << (8 * offset);
  unsigned int mask = 0xFF;
  mask = mask << (8 * offset);
  previous = previous & (~mask);
  return previous | next;
}

__device__ __forceinline__ unsigned short int
    __short_as_ushort(short int value) {
#ifdef __HIPCC__
  union {
    short int as_signed;
    unsigned short int as_unsigned;
  } val;
  val.as_signed = value;
  return val.as_unsigned;
#else
  unsigned short int result;
  asm("mov.b16 %0, %1;" : "=h"(result) : "h"(value));
  return result;
#endif
}

__device__ __forceinline__ short int
    __ushort_as_short(unsigned short int value) {
#ifdef __HIPCC__
  union {
    short int as_signed;
    unsigned short int as_unsigned;
  } val;
  val.as_unsigned = value;
  return val.as_signed;
#else
  short int result;
  asm("mov.b16 %0, %1;" : "=h"(result) : "h"(value));
  return result;
#endif
}

__device__ __forceinline__ unsigned int
    __hiloushort2uint(unsigned short int hi, unsigned short int lo) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    ushort2 as_short;
  } val;
  val.as_short.x = lo;
  val.as_short.y = hi;
  return val.as_int;
#else
  unsigned int result;
  asm("mov.b32 %0, {%1,%2};" : "=r"(result) : "h"(lo), "h"(hi));
  return result;
#endif
}

__device__ __forceinline__ unsigned short int
    __uint2loushort(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    ushort2 as_short;
  } val;
  val.as_int = value;
  return val.as_short.x;
#else
  unsigned short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return lo + 0 * hi;
#endif
}

__device__ __forceinline__ unsigned short int
    __uint2hiushort(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    ushort2 as_short;
  } val;
  val.as_int = value;
  return val.as_short.y;
#else
  unsigned short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return hi + 0 * lo;
#endif
}

__device__ __forceinline__ unsigned int __hiloshort2uint(short int hi,
                                                         short int lo) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_short.x = lo;
  val.as_short.y = hi;
  return val.as_int;
#else
  unsigned int result;
  asm("mov.b32 %0, {%1,%2};" : "=r"(result) : "h"(lo), "h"(hi));
  return result;
#endif
}

__device__ __forceinline__ short int __uint2loshort(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_int = value;
  return val.as_short.x;
#else
  short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return lo + 0 * hi;
#endif
}

__device__ __forceinline__ short int __uint2hishort(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_int = value;
  return val.as_short.y;
#else
  short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return hi + 0 * lo;
#endif
}

#ifdef LEGION_REDOP_HALF
__device__ __forceinline__ unsigned int __hilohalf2uint(__half hi, __half lo) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_short.x = __half_as_short(lo);
  val.as_short.y = __half_as_short(hi);
  return val.as_int;
#else
  unsigned int result;
  asm("mov.b32 %0, {%1,%2};"
      : "=r"(result)
      : "h"(__half_as_short(lo)), "h"(__half_as_short(hi)));
  return result;
#endif
}

__device__ __forceinline__ __half __uint2hihalf(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_int = value;
  return __short_as_half(val.as_short.y);
#else
  short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return __short_as_half(hi) + __half(0) * __short_as_half(lo);
#endif
}

__device__ __forceinline__ __half __uint2lohalf(unsigned int value) {
#ifdef __HIPCC__
  union {
    unsigned int as_int;
    short2 as_short;
  } val;
  val.as_int = value;
  return __short_as_half(val.as_short.x);
#else
  short int lo, hi;
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(value));
  return __short_as_half(lo) + __half(0) * __short_as_half(hi);
#endif
}
#endif

__device__ __forceinline__ unsigned int __int_as_uint(int value) {
#ifdef __HIPCC__
  union {
    int as_signed;
    unsigned int as_unsigned;
  } val;
  val.as_signed = value;
  return val.as_unsigned;
#else
  unsigned int result;
  asm("mov.b32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
#endif
}

__device__ __forceinline__ int __uint_as_int(unsigned int value) {
#ifdef __HIPCC__
  union {
    int as_signed;
    unsigned int as_unsigned;
  } val;
  val.as_unsigned = value;
  return val.as_signed;
#else
  int result;
  asm("mov.b32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
#endif
}

__device__ __forceinline__ unsigned long long
    __longlong_as_ulonglong(long long value) {
#ifdef __HIPCC__
  union {
    long long as_signed;
    unsigned long long as_unsigned;
  } val;
  val.as_signed = value;
  return val.as_unsigned;
#else
  unsigned long long result;
  asm("mov.b64 %0, %1;" : "=l"(result) : "l"(value));
  return result;
#endif
}

__device__ __forceinline__ long long
    __ulonglong_as_longlong(unsigned long long value) {
#ifdef __HIPCC__
  union {
    long long as_signed;
    unsigned long long as_unsigned;
  } val;
  val.as_unsigned = value;
  return val.as_signed;
#else
  long long result;
  asm("mov.b64 %0, %1;" : "=l"(result) : "l"(value));
  return result;
#endif
}

__device__ __forceinline__ double
    __ulonglong_as_double(unsigned long long value) {
#ifdef __HIPCC__
  union {
    unsigned long long as_int;
    double as_float;
  } val;
  val.as_int = value;
  return val.as_float;
#else
  double result;
  asm("mov.b64 %0, %1;" : "=d"(result) : "l"(value));
  return result;
#endif
}

__device__ __forceinline__ unsigned long long
    __double_as_ulonglong(double value) {
#ifdef __HIPCC__
  union {
    unsigned long long as_int;
    double as_float;
  } val;
  val.as_float = value;
  return val.as_int;
#else
  unsigned long long result;
  asm("mov.b64 %0, %1;" : "=l"(result) : "d"(value));
  return result;
#endif
}
#endif

template <typename T>
class SumReduction {
  // Empty definition
  // Specializations provided for each type
};

template <>
class SumReduction<bool> {
public:
  typedef bool LHS;
  typedef bool RHS;

  static constexpr bool identity = false;

  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void fold(RHS &rhs1, RHS rhs2);
};

template <>
class SumReduction<int32_t> {
public:
  typedef int32_t LHS;
  typedef int32_t RHS;

  static constexpr int32_t identity = 0;

  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void fold(RHS &rhs1, RHS rhs2);
};

template <>
class SumReduction<int64_t> {
public:
  typedef int64_t LHS;
  typedef int64_t RHS;

  static constexpr int64_t identity = 0;

  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void fold(RHS &rhs1, RHS rhs2);
};

template <>
class SumReduction<float> {
public:
  typedef float LHS;
  typedef float RHS;

  static constexpr float identity = 0.f;

  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void fold(RHS &rhs1, RHS rhs2);
};

template <>
class SumReduction<double> {
public:
  typedef double LHS;
  typedef double RHS;

  static constexpr double identity = 0.0;

  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __LEGION_CUDA_HD__ static void fold(RHS &rhs1, RHS rhs2);
};

template <>
__LEGION_CUDA_HD__ inline void SumReduction<bool>::apply<true>(LHS &lhs,
                                                               RHS rhs) {
  lhs = lhs || rhs;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<bool>::apply<false>(LHS &lhs,
                                                                RHS rhs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // GPU atomics need 4 byte alignment
  const uintptr_t unaligned = reinterpret_cast<uintptr_t>(&lhs);
  unsigned const offset = unaligned % sizeof(unsigned int);
  const uintptr_t aligned = unaligned - offset;
  unsigned int *ptr = reinterpret_cast<unsigned int *>(aligned);
  unsigned int newval = *ptr, oldval;
  do {
    RHS previous = __uint2bool(newval, offset);
    RHS next = previous || rhs;
    oldval = newval;
    newval = __bool2uint(newval, next, offset);
    newval = atomicCAS(ptr, oldval, newval);
  } while (oldval != newval);
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<LHS> atomic(lhs);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval || rhs;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic logical operations so use compare and swap
  TypePunning::Alias<int8_t, bool> oldval, newval;
  TypePunning::Pointer<int8_t> pointer((void *)&lhs);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() || rhs;
  } while (!__sync_bool_compare_and_swap(
      (int8_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<bool>::fold<true>(RHS &rhs1,
                                                              RHS rhs2) {
  rhs1 = rhs1 || rhs2;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<bool>::fold<false>(RHS &rhs1,
                                                               RHS rhs2) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // GPU atomics need 4 byte alignment
  const uintptr_t unaligned = reinterpret_cast<uintptr_t>(&rhs1);
  unsigned const offset = unaligned % sizeof(unsigned int);
  const uintptr_t aligned = unaligned - offset;
  unsigned int *ptr = reinterpret_cast<unsigned int *>(aligned);
  unsigned int newval = *ptr, oldval;
  do {
    RHS previous = __uint2bool(newval, offset);
    RHS next = previous || rhs2;
    oldval = newval;
    newval = __bool2uint(newval, next, offset);
    newval = atomicCAS(ptr, oldval, newval);
  } while (oldval != newval);
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<RHS> atomic(rhs1);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval || rhs2;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic logical operations so use compare and swap
  TypePunning::Alias<int8_t, bool> oldval, newval;
  TypePunning::Pointer<int8_t> pointer((void *)&rhs1);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() || rhs2;
  } while (!__sync_bool_compare_and_swap(
      (int8_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int32_t>::apply<true>(LHS &lhs,
                                                                  RHS rhs) {
  lhs += rhs;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int32_t>::apply<false>(LHS &lhs,
                                                                   RHS rhs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&lhs, rhs);
#else
  __sync_fetch_and_add(&lhs, rhs);
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int32_t>::fold<true>(RHS &rhs1,
                                                                 RHS rhs2) {
  rhs1 += rhs2;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int32_t>::fold<false>(RHS &rhs1,
                                                                  RHS rhs2) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&rhs1, rhs2);
#else
  __sync_fetch_and_add(&rhs1, rhs2);
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int64_t>::apply<true>(LHS &lhs,
                                                                  RHS rhs) {
  lhs += rhs;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int64_t>::apply<false>(LHS &lhs,
                                                                   RHS rhs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // Apparently there is no signed 64bit int atomic yet
  RHS newval = lhs, oldval;
  // Type punning like this is illegal in C++ but the
  // CUDA manual has an example just like it so fuck it
  unsigned long long int *ptr = (unsigned long long int *)&lhs;
  do {
    oldval = newval;
    newval += rhs;
    newval = __ulonglong_as_longlong(atomicCAS(
        ptr, __longlong_as_ulonglong(oldval), __longlong_as_ulonglong(newval)));
  } while (oldval != newval);
#else
  __sync_fetch_and_add(&lhs, rhs);
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int64_t>::fold<true>(RHS &rhs1,
                                                                 RHS rhs2) {
  rhs1 += rhs2;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<int64_t>::fold<false>(RHS &rhs1,
                                                                  RHS rhs2) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // Apparently there is no signed 64bit int atomic yet
  RHS newval = rhs1, oldval;
  // Type punning like this is illegal in C++ but the
  // CUDA manual has an example just like it so fuck it
  unsigned long long int *ptr = (unsigned long long int *)&rhs1;
  do {
    oldval = newval;
    newval += rhs2;
    newval = __ulonglong_as_longlong(atomicCAS(
        ptr, __longlong_as_ulonglong(oldval), __longlong_as_ulonglong(newval)));
  } while (oldval != newval);
#else
  __sync_fetch_and_add(&rhs1, rhs2);
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<float>::apply<true>(LHS &lhs,
                                                                RHS rhs) {
  lhs += rhs;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<float>::apply<false>(LHS &lhs,
                                                                 RHS rhs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&lhs, rhs);
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<LHS> atomic(lhs);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval + rhs;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic floating point operations so use compare and swap
  TypePunning::Alias<int32_t, float> oldval, newval;
  TypePunning::Pointer<int32_t> pointer((void *)&lhs);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() + rhs;
  } while (!__sync_bool_compare_and_swap(
      (int32_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<float>::fold<true>(RHS &rhs1,
                                                               RHS rhs2) {
  rhs1 += rhs2;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<float>::fold<false>(RHS &rhs1,
                                                                RHS rhs2) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&rhs1, rhs2);
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<RHS> atomic(rhs1);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval + rhs2;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic floating point operations so use compare and swap
  TypePunning::Alias<int32_t, float> oldval, newval;
  TypePunning::Pointer<int32_t> pointer((void *)&rhs1);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() + rhs2;
  } while (!__sync_bool_compare_and_swap(
      (int32_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<double>::apply<true>(LHS &lhs,
                                                                 RHS rhs) {
  lhs += rhs;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<double>::apply<false>(LHS &lhs,
                                                                  RHS rhs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if (__CUDA_ARCH__ >= 600) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&lhs, rhs);
#else
  RHS newval = lhs, oldval;
  // Type punning like this is illegal in C++ but the
  // CUDA manual has an example just like it so fuck it
  unsigned long long int *ptr = (unsigned long long int *)&lhs;
  do {
    oldval = newval;
    newval += rhs;
    newval = __ulonglong_as_double(atomicCAS(
        ptr, __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
  } while (oldval != newval);
#endif
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<LHS> atomic(lhs);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval + rhs;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic floating point operations so use compare and swap
  TypePunning::Alias<int64_t, double> oldval, newval;
  TypePunning::Pointer<int64_t> pointer((void *)&lhs);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() + rhs;
  } while (!__sync_bool_compare_and_swap(
      (int64_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<double>::fold<true>(RHS &rhs1,
                                                                RHS rhs2) {
  rhs1 += rhs2;
}

template <>
__LEGION_CUDA_HD__ inline void SumReduction<double>::fold<false>(RHS &rhs1,
                                                                 RHS rhs2) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if (__CUDA_ARCH__ >= 600) || defined(__HIP_DEVICE_COMPILE__)
  atomicAdd(&rhs1, rhs2);
#else
  RHS newval = rhs1, oldval;
  // Type punning like this is illegal in C++ but the
  // CUDA manual has an example just like it so fuck it
  unsigned long long int *ptr = (unsigned long long int *)&rhs1;
  do {
    oldval = newval;
    newval += rhs2;
    newval = __ulonglong_as_double(atomicCAS(
        ptr, __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
  } while (oldval != newval);
#endif
#else
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<RHS> atomic(rhs1);
  RHS oldval = atomic.load();
  RHS newval;
  do {
    newval = oldval + rhs2;
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  // No atomic floating point operations so use compare and swap
  TypePunning::Alias<int64_t, double> oldval, newval;
  TypePunning::Pointer<int64_t> pointer((void *)&rhs1);
  do {
    oldval.load(pointer);
    newval = oldval.as_two() + rhs2;
  } while (!__sync_bool_compare_and_swap(
      (int64_t *)pointer, oldval.as_one(), newval.as_one()));
#endif
#endif
}

// Follow the Legion style of defining reduction operators first and then
// adapting the interface for CUDA
template <typename T>
class AddCudaReductions : public T {
public:
  static bool const has_cuda_reductions = true;

  template <bool EXCLUSIVE>
  static __device__ void apply_cuda(typename T::LHS &lhs, typename T::RHS rhs) {
    T::template apply<EXCLUSIVE>(lhs, rhs);
  }

  template <bool EXCLUSIVE>
  static __device__ void fold_cuda(typename T::LHS &lhs, typename T::RHS rhs) {
    T::template fold<EXCLUSIVE>(lhs, rhs);
  }
};

void register_all_redops(Realm::Runtime rt) {
  // Registration is synchronous, so no need to capture events here
  rt.register_reduction<AddCudaReductions<SumReduction<bool>>>(
      get_realm_reduction_op_id_for_redop_id(redop_id_t::SUM_BOOL_REDOP_ID));
  rt.register_reduction<AddCudaReductions<SumReduction<int32_t>>>(
      get_realm_reduction_op_id_for_redop_id(redop_id_t::SUM_INT32_REDOP_ID));
  rt.register_reduction<AddCudaReductions<SumReduction<int64_t>>>(
      get_realm_reduction_op_id_for_redop_id(redop_id_t::SUM_INT64_REDOP_ID));
  rt.register_reduction<AddCudaReductions<SumReduction<float>>>(
      get_realm_reduction_op_id_for_redop_id(redop_id_t::SUM_FLOAT_REDOP_ID));
  rt.register_reduction<AddCudaReductions<SumReduction<double>>>(
      get_realm_reduction_op_id_for_redop_id(redop_id_t::SUM_DOUBLE_REDOP_ID));
}

} // namespace FlexFlow
