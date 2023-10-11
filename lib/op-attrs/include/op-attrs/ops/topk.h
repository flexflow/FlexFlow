#ifndef _FLEXFLOW_TOPK_ATTRS_H
#define _FLEXFLOW_TOPK_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

//I think we should add axis 
//pytorch code: torch.topk(input_tensor, k, largest=True, sorted=True, dim=dim)
struct TopKAttrs {
  req<int> k;
  req<bool> sorted;
  req<int> axis;
  bool is_valid(ParallelTensorShape const &) const;
};
FF_VISITABLE_STRUCT(TopKAttrs, k, sorted,axis);
CHECK_VALID_OP_ATTR(TopKAttrs);

ParallelTensorShape get_output_shape(TopKAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
