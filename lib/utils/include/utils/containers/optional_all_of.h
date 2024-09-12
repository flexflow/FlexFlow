#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_OPTIONAL_ALL_OF_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_OPTIONAL_ALL_OF_H

#include <optional>

namespace FlexFlow {

template <typename Container, typename Function>
std::optional<bool> optional_all_of(Container const &container,
                                    Function const &func) {
  for (auto const &element : container) {
    std::optional<bool> condition = func(element);
    if (!condition.has_value()) {
      return std::nullopt;
    }

    if (!condition.value()) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

#endif
