#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReverseAttrs {
  ff_dim_t axis;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(ReverseAttrs, axis);
CHECK_VALID_OP_ATTR(ReverseAttrs);

ParallelTensorShape get_output_shape(ReverseAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
