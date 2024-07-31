#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_KEY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_KEY_H

namespace FlexFlow {

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k) {
  return m.find(k) != m.end();
}

} // namespace FlexFlow

#endif
