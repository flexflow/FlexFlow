#ifndef _FLEXFLOW_COMBINE_ATTRS_H
#define _FLEXFLOW_COMBINE_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct CombineAttrs {
  ff_dim_t combine_dim;
  req<int> combine_degree;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(CombineAttrs, combine_dim, combine_degree);
CHECK_VALID_OP_ATTR(CombineAttrs);

ParallelTensorShape get_output_shape(CombineAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
