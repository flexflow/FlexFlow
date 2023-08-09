#ifndef _FLEXFLOW_UTILS_INCLUDE_UNIQUE_H
#define _FLEXFLOW_UTILS_INCLUDE_UNIQUE_H

#include <memory>

namespace FlexFlow {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace FlexFlow

#endif
