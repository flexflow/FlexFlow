#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GatherAttrs {
  ff_dim_t dim;
};
FF_VISITABLE_STRUCT(GatherAttrs, dim);
CHECK_VALID_OP_ATTR(GatherAttrs);

std::vector<ParallelTensorShape> get_output_shapes(GatherAttrs const &,
                                                   ParallelTensorShape const &,
                                                   ParallelTensorShape const &);
} // namespace FlexFlow

#endif
