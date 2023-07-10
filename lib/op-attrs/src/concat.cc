#include "op-attrs/ops/concat.h"

namespace FlexFlow {

ConcatAttrs::ConcatAttrs(ff_dim_t _axis) : axis(_axis) {}

/* bool ConcatAttrs::is_valid( */
/*     std::vector<ParallelTensorShape> const &input) const { */
/*   bool valid = true; */
/*   for (auto p : input) { */
/*     valid &= p.is_valid(); */
/*   } */
/*   return valid; */
/* } */

} // namespace FlexFlow
