#ifndef _UTILS_INCLUDE_UTILS_TYPE_INDEX_H
#define _UTILS_INCLUDE_UTILS_TYPE_INDEX_H

#include "fmt.h"
#include <typeindex>

namespace FlexFlow {

template <typename T>
std::type_index type_index() {
  return std::type_index(typeid(T));
}

template <typename T>
bool matches(std::type_index idx) {
  return idx == type_index<T>();
}

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::std::type_index> : formatter<std::string> {
  template <typename FormatContext>
  auto format(std::type_index const &type_idx, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(type_idx.name(), ctx);
  }
};

} // namespace fmt

#endif
