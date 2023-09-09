#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_FLATMAP_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_FLATMAP_H

#include <sstream>
#include <vector>
#include <type_traits>

namespace FlexFlow {

template <typename In, typename F>
auto flatmap(std::vector<In> const &v, F const &f) {
  std::vector<std::invoke_result_t<F, In>> result;
  for (In const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename F,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<std::invoke_result_t<F, char>>,
                             std::string>>>
std::string flatmap(std::string const &s, F const &f) {
  std::ostringstream oss;
  for (char c : s) {
    oss << f(c);
  }
  return oss.str();
}

} // namespace FlexFlow

#endif
