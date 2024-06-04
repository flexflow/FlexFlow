#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

size_t num_dims(TensorShape const &);
size_t dim_at_idx(TensorShape const &, ff_dim_t);
size_t &dim_at_idx(TensorShape &, ff_dim_t);

} // namespace FlexFlow

#endif
