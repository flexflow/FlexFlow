#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_VALUE_TYPE_H

#include <fmt/format.h>
#include <functional>
#include <libassert/assert.hpp>
#include <nlohmann/json.hpp>
#include <ostream>
#include <sstream>

namespace FlexFlow {

template <int TAG>
struct value_type {
  value_type() = delete;

  value_type(value_type const &) {
    PANIC();
  }
  value_type &operator=(value_type const &) {
    PANIC();
  }

  value_type(value_type &&) {
    PANIC();
  }
  value_type &operator=(value_type &&) {
    PANIC();
  }

  bool operator==(value_type const &) const {
    PANIC();
  }
  bool operator!=(value_type const &) const {
    PANIC();
  }
};

template <int TAG>
std::string format_as(value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
void to_json(nlohmann::json &, value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
std::ostream &operator<<(std::ostream &s, value_type<TAG> const &x) {
  PANIC();
}

} // namespace FlexFlow

namespace std {

template <int TAG>
struct hash<::FlexFlow::value_type<TAG>> {
  size_t operator()(::FlexFlow::value_type<TAG> const &) const {
    PANIC();
  };
};

} // namespace std

#endif
