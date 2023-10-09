#ifndef _FLEXFLOW_REDUCTION_ATTRS_H
#define _FLEXFLOW_REDUCTION_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReductionAttrs {
  ff_dim_t reduction_dim;
  req<int> reduction_degree;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(ReductionAttrs, reduction_dim, reduction_degree);
CHECK_VALID_OP_ATTR(ReductionAttrs);

ParallelTensorShape get_output_shape(ReductionAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
