#ifndef _UTILS_INCLUDE_UTILS_TYPE_INDEX_H
#define _UTILS_INCLUDE_UTILS_TYPE_INDEX_H

#include <typeindex>

namespace FlexFlow {

template <typename T>
std::type_index type_index() {
  return std::type_index(typeid(T));
}

} // namespace FlexFlow

#endif
