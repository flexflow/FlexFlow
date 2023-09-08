#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H

#include "utils/expected.h"
#include <algorithm>
#include <new>

namespace FlexFlow {

template <typename ErrorCodeType,
          ErrorCodeType StatusOK,
          ErrorCodeType UnexpectedNull,
          ErrorCodeType AllocationFailed>
struct LibraryUtils {
  template <typename T>
  using err = expected<ErrorCodeType, T>;

  template <typename T>
  static err<T *> allocate_opaque(T const &t) {
    T *ptr = new (std::nothrow) T(t);
    if (ptr == nullptr) {
      return StatusOK;
    }
    return ptr;
  }

  template <typename T>
  static err<T> allocate_opaque(T &&t) {
    T *ptr = new (std::nothrow) T(std::move(t));
    if (ptr == nullptr) {
      return AllocationFailed;
    }
    return ptr;
  }

  template <
      typename Opaque,
      typename Unwrapped = decltype(*unwrap_opaque(std::declval<Opaque>()))>
  static err<Opaque> new_opaque(Unwrapped const &f) {
    return allocate_opaque<Opaque>(f).map([](Unwrapped *ptr) { return ptr; });
  }

  template <typename T>
  static ErrorCodeType output_stored(T const &t, T *out) {
    return output_stored(new_opaque(t), out);
  }

  template <typename T>
  static ErrorCodeType output_stored(err<T> const &e, T *out) {
    if (e.has_value()) {
      *out = e.value();
      return StatusOK;
    } else {
      return e.error();
    }
  }

  template <typename T>
  ErrorCodeType deallocate_opaque(T const &opaque) {
    auto unwrapped = unwrap_opaque(opaque);
    if (unwrapped == nullptr) {
      return UnexpectedNull;
    }

    delete unwrapped;
    return StatusOK;
  }
};

} // namespace FlexFlow

#endif
