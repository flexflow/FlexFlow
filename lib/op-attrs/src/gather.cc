#include "op-attrs/ops/gather.h"

namespace FlexFlow {

GatherAttrs::GatherAttrs(ff_dim_t _dim) : dim(_dim) {}

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
