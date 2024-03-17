#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_GET_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_GET_H

#include "utils/ff_exceptions/ff_exceptions.h"
#include "visit.h"
#include <cstddef>
#include <sstream>
#include <tuple>
#include <variant>

namespace FlexFlow {

template <typename... Types>
std::variant<Types...> get(std::tuple<Types...> const &t, int idx) {
  std::size_t tuple_size = std::tuple_size<decltype(t)>::value;
  if (idx < 0 || idx >= tuple_size) {
    std::ostringstream oss;
    oss << "Error: idx " << idx << " out of bounds for tuple of size "
        << tuple_size;
    throw mk_runtime_error(oss.str());
  }

  std::variant<Types...> result;
  visit_tuple(t, [&](int i, auto const &val) {
    if (i == idx) {
      result = val;
    }
  });
  return result;
}

} // namespace FlexFlow

#endif
