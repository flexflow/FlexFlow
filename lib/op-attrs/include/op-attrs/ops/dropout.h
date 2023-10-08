#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DropoutAttrs {
  req<float> rate;
  req<unsigned long long> seed;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(DropoutAttrs, rate, seed);
CHECK_VALID_OP_ATTR(DropoutAttrs);

ParallelTensorShape get_output_shape(DropoutAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
