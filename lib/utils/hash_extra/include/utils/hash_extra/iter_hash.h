#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_ITER_HASH_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_ITER_HASH_H

#include <cstddef>
#include <iterator>
#include "utils/hash_extra/hash_combine.h"

namespace FlexFlow {

template <typename It>
void iter_hash(std::size_t &hash_val, It start, It end) {
  hash_combine(hash_val, std::distance(start, end));
  for (; start < end; start++) {
    hash_combine(hash_val, *start);
  }
}

} // namespace FlexFlow

#endif
