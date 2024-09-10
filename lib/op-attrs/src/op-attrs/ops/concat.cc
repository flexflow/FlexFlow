#include "op-attrs/ops/concat.h"

namespace FlexFlow {

/* bool ConcatAttrs::is_valid( */
/*     std::vector<ParallelTensorShape> const &input) const { */
/*   bool valid = true; */
/*   for (auto p : input) { */
/*     valid &= p.is_valid(); */
/*   } */
/*   return valid; */
/* } */

TensorShape get_output_shape(ConcatAttrs const &,
                             std::vector<TensorShape> const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
