#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

size_t num_dims(TensorShape const &);
size_t dim_at_idx(TensorShape const &, ff_dim_t);
size_t &dim_at_idx(TensorShape &, ff_dim_t);
size_t get_num_elements(TensorShape const &);
size_t get_size_in_bytes(TensorShape const &);

bool tensor_shape_is_broadcastable_to(TensorShape const &curr, TensorShape const &goal);
std::optional<TensorShape> get_broadcast_target_shape(std::unordered_set<TensorShape> const &);

} // namespace FlexFlow

#endif
