#include "op-attrs/ops/gather.h"

namespace FlexFlow {

TensorShape get_output_shape(GatherAttrs const &,
                             TensorShape const &input,
                             TensorShape const &index) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index) {
  NOT_IMPLEMENTED();
}

/* bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
 * ParallelTensorShape const &rhs) const { */
/*   if (lhs.num_dims() != rhs.num_dims()) { */
/*     return false; */
/*   } */
/*   for (int i = 0; i < lhs.num_dims(); i++) { */
/*     if (i != this->legion_dim && */
/*         lhs.at(i).size < rhs.at(i).size) { */
/*       return false; */
/*     } */
/*   } */
/*   return true; */
/* } */

} // namespace FlexFlow
