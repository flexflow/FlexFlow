#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_MAP2_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_MAP2_H

#include "utils/containers/contains_key.h"
#include "utils/containers/get_element_type.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/vector_transform.h"
#include "utils/type_traits_core.h"
#include <libassert/assert.hpp>
#include <map>

namespace FlexFlow {

template <typename KF,
          typename VF,
          typename C,
          typename K = std::invoke_result_t<KF, get_element_type_t<C>>,
          typename V = std::invoke_result_t<VF, get_element_type_t<C>>>
std::map<K, V> generate_map2(C const &c, KF &&kf, VF &&vf) {
  static_assert(is_lt_comparable_v<K>,
                "Key type should be ordered (but is not)");

  std::map<K, V> result;
  for (auto const &x : c) {
    K k = kf(x);
    V v = vf(x);

    if (contains_key(result, k)) {
      ASSERT(result.at(k) == v);
    }

    result.insert(std::pair{k, v});
  }

  return result;
}

} // namespace FlexFlow

#endif
