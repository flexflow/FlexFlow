#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ConcatAttrs {
  ff_dim_t axis;
  bool is_valid(std::vector<ParallelTensorShape> const &input) const;
};

ParallelTensorShape get_output_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &);

FF_VISITABLE_STRUCT(ConcatAttrs, axis);
CHECK_VALID_OP_ATTR(ConcatAttrs);

} // namespace FlexFlow

#endif
