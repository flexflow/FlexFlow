#include "realm-execution/redops/realm_redop_registry.h"
#include "realm-execution/redops/redop_id_t.dtg.h"
#include <cassert>
#include <realm.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <realm/cuda/cuda_redop.h>
#endif

namespace FlexFlow {

// Reduction operators and related infrastructure borrowed from Legion. We
// maintain the Legion naming scheme to maximizing compatibility with the
// existing code, despite not otherwise relying or using Legion in any way.
// https://gitlab.com/StanfordLegion/legion/-/blob/5263aeff477fb94239c50d9306d58c4244e9fc38/runtime/legion/api/redop.inl#L31
#if !defined(__cpp_lib_atomic_ref) || (__cpp_lib_atomic_ref < 201806L)
namespace TypePunning {
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
// Legion's non-exclusive bool/int64 reduction paths (borrowed below) assume
// these helpers exist, but they aren't real CUDA/HIP builtins, so we provide
// them ourselves.
//
// __longlong_as_ulonglong/__ulonglong_as_longlong: signed<->unsigned 64-bit
// reinterpretation (needed because atomicCAS only takes unsigned long long).
__device__ __forceinline__ unsigned long long int
    __longlong_as_ulonglong(long long int v) {
  return static_cast<unsigned long long int>(v);
}
__device__ __forceinline__ long long int
    __ulonglong_as_longlong(unsigned long long int v) {
  return static_cast<long long int>(v);
}

// __uint2bool/__bool2uint: read/write a single bool packed into one byte of
// a 4-byte-aligned word, at byte `offset` within that word (needed because
// GPU atomicCAS requires 4-byte alignment, but a lone bool doesn't have
// one).
__device__ __forceinline__ bool __uint2bool(unsigned int word,
                                            unsigned int offset) {
  return static_cast<bool>((word >> (offset * 8)) & 0xFFu);
}
__device__ __forceinline__ unsigned int
    __bool2uint(unsigned int word, bool value, unsigned int offset) {
  unsigned int const shift = offset * 8;
  unsigned int const mask = 0xFFu << shift;
  unsigned int const byte_val = (static_cast<unsigned int>(value) & 0xFFu)
                                << shift;
  return (word & ~mask) | byte_val;
}

// apply_cuda/fold_cuda are identical (just delegate to apply/fold) for every
// SumReduction<T> specialization, so factored into a macro rather than
// repeated per type. Required by Realm::Cuda::add_cuda_redop_kernels to
// build a GPU-resident reduction kernel for this redop.
#define __LEGION_CUDA_REDOP_METHODS__                                          \
  template <bool EXCLUSIVE>                                                    \
  __device__ static void apply_cuda(LHS &lhs, RHS rhs) {                       \
    apply<EXCLUSIVE>(lhs, rhs);                                                \
  }                                                                            \
  template <bool EXCLUSIVE>                                                    \
  __device__ static void fold_cuda(RHS &rhs1, RHS rhs2) {                      \
    fold<EXCLUSIVE>(rhs1, rhs2);                                               \
  }
#else
#define __LEGION_CUDA_REDOP_METHODS__
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
  __LEGION_CUDA_REDOP_METHODS__
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
  __LEGION_CUDA_REDOP_METHODS__
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
  __LEGION_CUDA_REDOP_METHODS__
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
  __LEGION_CUDA_REDOP_METHODS__
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
  __LEGION_CUDA_REDOP_METHODS__
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

namespace {
template <typename REDOP>
void register_sum_redop(::Realm::Runtime &rt, ::Realm::ReductionOpID id) {
  ::Realm::ReductionOpUntyped *redop =
      ::Realm::ReductionOpUntyped::create_reduction_op<REDOP>();
#if defined(__CUDACC__) || defined(__HIPCC__)
  ::Realm::Cuda::add_cuda_redop_kernels<REDOP>(redop);
#endif
  bool ok = rt.register_reduction(id, redop);
  assert(ok);
}

} // namespace

void register_all_redops() {
  ::Realm::Runtime rt = ::Realm::Runtime::get_runtime();
  // Registration is synchronous, so no need to capture events here
  register_sum_redop<SumReduction<bool>>(
      rt, static_cast<::Realm::ReductionOpID>(redop_id_t::SUM_BOOL_REDOP_ID));
  register_sum_redop<SumReduction<int32_t>>(
      rt, static_cast<::Realm::ReductionOpID>(redop_id_t::SUM_INT32_REDOP_ID));
  register_sum_redop<SumReduction<int64_t>>(
      rt, static_cast<::Realm::ReductionOpID>(redop_id_t::SUM_INT64_REDOP_ID));
  register_sum_redop<SumReduction<float>>(
      rt, static_cast<::Realm::ReductionOpID>(redop_id_t::SUM_FLOAT_REDOP_ID));
  register_sum_redop<SumReduction<double>>(
      rt, static_cast<::Realm::ReductionOpID>(redop_id_t::SUM_DOUBLE_REDOP_ID));
}

} // namespace FlexFlow
