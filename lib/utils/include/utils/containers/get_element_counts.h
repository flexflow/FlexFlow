#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H

#include <string>
#include <unordered_map> 
#include <vector>
#include "utils/containers/contains_key.h"

namespace FlexFlow {

template <typename T>
std::unordered_map<T, int> get_element_counts(std::vector<T> const &v) {
  std::unordered_map<T, int> counts;
  for (T const &t : v) {
    if (!contains_key(counts, t)) {
      counts[t] = 0;
    }
    counts.at(t)++;
  }
  return counts;
}

std::unordered_map<char, int> get_element_counts(std::string const &);

} // namespace FlexFlow

#endif
