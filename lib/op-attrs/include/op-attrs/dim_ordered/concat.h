#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_CONCAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_CONCAT_H

#include "op-attrs/dim_ordered.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> concat(FFOrdered<T> const &l, FFOrdered<T> const &r) {
  std::vector<T> l_vec = std::vector(l.cbegin(), l.cend());
  std::vector<T> r_vec = std::vector(r.cbegin(), r.cend());

  std::vector<T> raw_result = concat_vectors(l_vec, r_vec);

  return FFOrdered<T>(raw_result.cbegin(), raw_result.cend());
}

template <typename T>
FFOrdered<T> concat(std::vector<FFOrdered<T>> const &inputs) {
  std::vector<std::vector<T>> vec_inputs = transform(inputs, 
                                                     [](FFOrdered<T> const &input) {
                                                       return std::vector<T>(input.cbegin(), input.cend());
                                                     });

  std::vector<T> raw_result = concat_vectors(vec_inputs);

  return FFOrdered<T>(raw_result.cbegin(), raw_result.cend());
}

} // namespace FlexFlow

#endif
