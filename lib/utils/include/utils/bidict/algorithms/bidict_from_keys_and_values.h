#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_KEYS_AND_VALUES_H

#include "utils/containers/zip.h"
#include "utils/bidict/bidict.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> bidict_from_keys_and_values(std::vector<L> const &ls, std::vector<R> const &rs) {
  size_t l_size = ls.size();
  size_t r_size = rs.size();
  if (l_size != r_size) {
    throw mk_runtime_error(fmt::format("__FUNC__ recieved keys (of size {}) not matching values (of size {})", l_size, r_size));
  }

  return bidict_from_pairs(zip(ls, rs));
}

} // namespace FlexFlow

#endif
