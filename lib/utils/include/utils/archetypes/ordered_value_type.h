#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_ORDERED_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_ORDERED_VALUE_TYPE_H

#include <functional>
#include <libassert/assert.hpp>
#include <nlohmann/json.hpp>

namespace FlexFlow {

template <int TAG>
struct ordered_value_type {
  ordered_value_type() = delete;

  ordered_value_type(ordered_value_type const &) {
    PANIC();
  }
  ordered_value_type &operator=(ordered_value_type const &) {
    PANIC();
  }

  ordered_value_type(ordered_value_type &&) {
    PANIC();
  }
  ordered_value_type &operator=(ordered_value_type &&) {
    PANIC();
  }

  bool operator==(ordered_value_type const &) const {
    PANIC();
  }
  bool operator!=(ordered_value_type const &) const {
    PANIC();
  }

  bool operator<(ordered_value_type const &) const {
    PANIC();
  }
  bool operator>(ordered_value_type const &) const {
    PANIC();
  }
  bool operator<=(ordered_value_type const &) const {
    PANIC();
  }
  bool operator>=(ordered_value_type const &) const {
    PANIC();
  }
};

template <int TAG>
std::string format_as(ordered_value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
void to_json(nlohmann::json &, ordered_value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
std::ostream &operator<<(std::ostream &s, ordered_value_type<TAG> const &x) {
  PANIC();
}

} // namespace FlexFlow

namespace std {

template <int TAG>
struct hash<::FlexFlow::ordered_value_type<TAG>> {
  size_t operator()(::FlexFlow::ordered_value_type<TAG> const &) const {
    PANIC();
  };
};

} // namespace std

#endif
