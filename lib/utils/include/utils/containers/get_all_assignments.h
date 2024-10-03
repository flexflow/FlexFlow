#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_ASSIGNMENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_ASSIGNMENTS_H

#include <unordered_set>
#include <unordered_map>
#include "utils/containers/vector_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_map_from_pairs.h"
#include "utils/containers/keys.h"
#include "utils/containers/zip.h"
#include "utils/containers/cartesian_product.h"
#include "utils/hash/unordered_map.h"
#include <vector>

namespace FlexFlow {

/**
 * @note If \p options_per_key is empty, an set containing a single empty assignment is returned 
 */
template <typename K, typename V>
std::unordered_set<std::unordered_map<K, V>> get_all_assignments(std::unordered_map<K, std::unordered_set<V>> const &options_per_key) {
  if (options_per_key.empty()) {
    return {{}};
  }

  std::vector<K> ordered_keys = vector_of(keys(options_per_key));
  std::vector<std::unordered_set<V>> ordered_value_option_sets = transform(ordered_keys,
                                                                           [&](K const &k) { return options_per_key.at(k); });

  std::unordered_set<std::unordered_map<K, V>> result = transform(
    cartesian_product(ordered_value_option_sets),
    [&](std::vector<V> const &chosen_values) {
      return unordered_map_from_pairs(zip(ordered_keys, chosen_values));
    });

  return result;
}

} // namespace FlexFlow

#endif
