#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ENUMERATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ENUMERATE_H

#include "op-attrs/dim_ordered.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

/**
 * @brief Generate a map from indices to elements of \p c.
 *
 * @note We return a <tt>std::map</tt> to prevent mixups of \ref ff_dim_t and
 * \ref legion_dim_t. Note that <tt>std::map</tt> provides ordered iteration in
 * increasing order, so iterating through the result of this function should
 * function as expected.
 */
template <typename T>
std::map<ff_dim_t, T> enumerate(FFOrdered<T> const &ff_ordered) {
  std::map<ff_dim_t, T> result;
  for (int raw_ff_dim : count(ff_ordered.size())) {
    ff_dim_t ff_dim = ff_dim_t{raw_ff_dim};
    result.insert({ff_dim, ff_ordered.at(ff_dim)});
  }
  return result;
}

} // namespace FlexFlow

#endif
