#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_RAPIDCHECKABLE_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_RAPIDCHECKABLE_VALUE_TYPE_H

#include <fmt/format.h>
#include <functional>
#include <libassert/assert.hpp>
#include <nlohmann/json.hpp>
#include <ostream>
#include <rapidcheck.h>
#include <sstream>

namespace FlexFlow {

template <int TAG>
struct rapidcheckable_value_type {
  rapidcheckable_value_type() = delete;

  rapidcheckable_value_type(rapidcheckable_value_type const &) {
    PANIC();
  }
  rapidcheckable_value_type &operator=(rapidcheckable_value_type const &) {
    PANIC();
  }

  rapidcheckable_value_type(rapidcheckable_value_type &&) {
    PANIC();
  }
  rapidcheckable_value_type &operator=(rapidcheckable_value_type &&) {
    PANIC();
  }

  bool operator==(rapidcheckable_value_type const &) const {
    PANIC();
  }
  bool operator!=(rapidcheckable_value_type const &) const {
    PANIC();
  }
  bool operator<(rapidcheckable_value_type const &) const {
    PANIC();
  }
};

template <int TAG>
std::string format_as(rapidcheckable_value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
void to_json(nlohmann::json &, rapidcheckable_value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
std::ostream &operator<<(std::ostream &s,
                         rapidcheckable_value_type<TAG> const &x) {
  PANIC();
}

} // namespace FlexFlow

namespace rc {

template <int TAG>
struct Arbitrary<::FlexFlow::rapidcheckable_value_type<TAG>> {
  static Gen<::FlexFlow::rapidcheckable_value_type<TAG>> arbitrary() {
    PANIC();
  }
};

} // namespace rc

namespace std {

template <int TAG>
struct hash<::FlexFlow::rapidcheckable_value_type<TAG>> {
  size_t operator()(::FlexFlow::rapidcheckable_value_type<TAG> const &) const {
    PANIC();
  };
};

} // namespace std

#endif
