#ifndef _FLEXFLOW_REPLICATE_ATTRS_H
#define _FLEXFLOW_REPLICATE_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReplicateAttrs {
  ff_dim_t replicate_dim;
  req<int> replicate_degree;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(ReplicateAttrs, replicate_dim, replicate_degree);
CHECK_VALID_OP_ATTR(ReplicateAttrs);

ParallelTensorShape get_output_shape(ReplicateAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
