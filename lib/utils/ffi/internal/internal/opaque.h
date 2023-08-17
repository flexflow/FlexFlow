#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_OPAQUE_H

#include "error.h"
#include "flexflow/utils.h"
#include "utils/containers.h"
#include "utils/expected.h"
#include <algorithm>
#include <new>
#include <vector>

template <typename T>
struct opaque_to_underlying;

template <typename T>
struct underlying_to_opaque;

template <typename T>
using opaque_to_underlying_t = typename opaque_to_underlying<T>::type;

template <typename T>
using underlying_to_opaque_t = typename underlying_to_opaque<T>::type;

template <typename Opaque>
opaque_to_underlying_t<Opaque> *unwrap_opaque(Opaque const &opaque) {
  if (opaque.impl == nullptr) {
    throw make_utils_exception(
        FLEXFLOW_UTILS_UNEXPECTED_NULLPTR_IN_OPAQUE_HANDLE);
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

template <typename Opaque>
std::vector<opaque_to_underlying_t<Opaque>>
    c_deref_opaque_list(Opaque const *start, size_t num_elements) {
  std::vector<Opaque> exp_preds_vector =
      transform(start, start + num_elements, [](Opaque const &t) {
        return c_deref_opaque(t);
      });
}

#define REGISTER_OPAQUE(OPAQUE, UNDERLYING)                                    \
  template <>                                                                  \
  struct opaque_to_underlying<OPAQUE> {                                        \
    using type = UNDERLYING;                                                   \
  };                                                                           \
  template <>                                                                  \
  struct underlying_to_opaque<UNDERLYING> {                                    \
    using type = OPAQUE;                                                       \
  };

template <typename Opaque, typename... Args>
Opaque new_opaque(Args &&...args) {
  using Underlying = opaque_to_underlying_t<Opaque>;

  Underlying *ptr = new (std::nothrow) Underlying(std::forward<Args>(args)...);
  if (ptr == nullptr) {
    throw make_utils_exception(FLEXFLOW_UTILS_ALLOCATION_FAILED);
  }
  return Opaque{ptr};
}

template <typename Underlying>
underlying_to_opaque_t<Underlying> new_opaque(Underlying const &underlying) {
  return new_opaque<underlying_to_opaque_t<Underlying>>(underlying);
}

template <typename Opaque>
void delete_opaque(Opaque const &opaque) {
  using Underlying = opaque_to_underlying_t<Opaque>;

  Underlying *underlying = unwrap_opaque(opaque);
  if (underlying == nullptr) {
    throw make_utils_exception(
        FLEXFLOW_UTILS_UNEXPECTED_NULLPTR_IN_OPAQUE_HANDLE);
  }

  delete underlying;
}

#endif
