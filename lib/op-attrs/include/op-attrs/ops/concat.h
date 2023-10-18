#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ConcatAttrs {
  ff_dim_t axis;
  req<int> num_inputs;
};
FF_VISITABLE_STRUCT(ConcatAttrs, axis, num_inputs);
CHECK_VALID_OP_ATTR(ConcatAttrs);

ParallelTensorShape get_output_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &);
} // namespace FlexFlow

#endif
