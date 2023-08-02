#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H

#include "utils/expected.h"
#include <algorithm>
#include <new>
#include "flexflow/utils.h"
#include "error.h"

template <typename T>
struct opaque_to_underlying;

template <typename T>
struct underlying_to_opaque;

template <typename T>
struct internal_to_external;

template <typename T>
struct external_to_internal;

template <typename T> 
struct enum_mapping;

template <typename T>
using opaque_to_underlying_t = typename opaque_to_underlying<T>::type;

template <typename T>
using underlying_to_opaque_t = typename underlying_to_opaque<T>::type;

template <typename T>
using internal_to_external_t = typename internal_to_external<T>::type;

template <typename T>
using external_to_internal_t = typename external_to_internal<T>::type;

template <typename Opaque>
opaque_to_underlying_t<Opaque> *unwrap_opaque(Opaque const &opaque) {
  if (opaque.impl == nullptr) {
    throw make_utils_exception(FLEXFLOW_UTILS_UNEXPECTED_NULLPTR_IN_OPAQUE_HANDLE);
  }

  return static_cast<opaque_to_underlying_t<Opaque> *>(opaque.impl);
}

template <typename Opaque>
opaque_to_underlying_t<Opaque> const *c_unwrap_opaque(Opaque const &opaque) {
  return unwrap_opaque(opaque);
}

template <typename Opaque>
opaque_to_underlying_t<Opaque> const &c_deref_opaque(Opaque const &opaque) {
  return *unwrap_opaque(opaque);
}

template <typename Opaque>
opaque_to_underlying_t<Opaque> &deref_opaque(Opaque const &opaque) {
  return *unwrap_opaque(opaque);
}

#define REGISTER_OPAQUE(OPAQUE, UNDERLYING) \
  template <> \
  struct opaque_to_underlying<OPAQUE> { \
    using type = UNDERLYING; \
  }; \
  template <> \
  struct underlying_to_opaque<UNDERLYING> { \
    using type = OPAQUE; \
  }; 

#define REGISTER_FFI_ENUM(EXTERNAL, INTERNAL, ERROR_CODE, ...) \
  template <> \
  struct external_to_internal<EXTERNAL> { \
    using type = INTERNAL; \
  }; \
  template <> \
  struct internal_to_external<INTERNAL> { \
    using type = EXTERNAL; \
  }; \
  template <> \
  struct enum_mapping<EXTERNAL> { \
    static const bidict<EXTERNAL, INTERNAL> mapping; \
    static constexpr decltype(ERROR_CODE) err_code = ERROR_CODE; \
  }; \
  const bidict<EXTERNAL, INTERNAL> enum_mapping<EXTERNAL>::mapping = __VA_ARGS__;


template <typename Opaque, typename... Args>
Opaque new_opaque(Args &&... args) {
  using Underlying = opaque_to_underlying_t<Opaque>;

  Underlying *ptr = new (std::nothrow) Underlying(std::forward<Args>(args)...);
  if (ptr == nullptr) {
    throw make_utils_exception(FLEXFLOW_UTILS_ALLOCATION_FAILED);
  }
  return Opaque{ptr};
}

template <typename Opaque>
void delete_opaque(Opaque const &opaque) {
  using Underlying = opaque_to_underlying_t<Opaque>;

  Underlying *underlying = unwrap_opaque(opaque);
  if (underlying == nullptr) {
    throw make_utils_exception(FLEXFLOW_UTILS_UNEXPECTED_NULLPTR_IN_OPAQUE_HANDLE);
  }

  delete underlying;
}

#endif
