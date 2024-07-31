#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONCAT_VECTORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONCAT_VECTORS_H

#include "utils/containers/extend_vector.h"

namespace FlexFlow {

template <typename T>
std::vector<T> concat_vectors(std::vector<T> const &prefix,
                              std::vector<T> const &postfix) {
  std::vector<T> result = prefix;
  extend_vector(result, postfix);
  return result;
}

template <typename T>
std::vector<T> concat_vectors(std::vector<std::vector<T>> const &vecs) {
  std::vector<T> result;
  for (std::vector<T> const &v : vecs) {
    extend_vector(result, v);
  }
  return result;
}

} // namespace FlexFlow

#endif
