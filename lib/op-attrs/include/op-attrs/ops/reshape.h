#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReshapeAttrs {
  TensorShape shape;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(ReshapeAttrs, shape);
CHECK_VALID_OP_ATTR(ReshapeAttrs);

ParallelTensorShape get_output_shape(ReshapeAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
