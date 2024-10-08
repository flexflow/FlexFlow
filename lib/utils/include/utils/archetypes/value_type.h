#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_VALUE_TYPE_H

#include <cassert>
#include <functional>

namespace FlexFlow {

template <int TAG>
struct value_type {
  value_type() = delete;

  value_type(value_type const &) {
    assert(false);
  }
  value_type &operator=(value_type const &) {
    assert(false);
  }

  value_type(value_type &&) {
    assert(false);
  }
  value_type &operator=(value_type &&) {
    assert(false);
  }

  bool operator==(value_type const &) const {
    assert(false);
  }
  bool operator!=(value_type const &) const {
    assert(false);
  }
};

} // namespace FlexFlow

namespace std {

template <int TAG>
struct hash<::FlexFlow::value_type<TAG>> {
  size_t operator()(::FlexFlow::value_type<TAG> const &) const {
    assert(false);
  };
};

} // namespace std

#endif
