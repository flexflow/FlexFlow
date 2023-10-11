#include "op-attrs/ops/gather.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
                           ParallelTensorShape const &rhs) const {
  if (lhs.dims.num_dims() != rhs.dims.num_dims()) {
    return false;
  }
  for (auto i : lhs.dims) {
    if (ff_dim_t(i.size) != this->dim &&
        lhs.at(ff_dim_t(i.size)).size < rhs.at(ff_dim_t(i.size)).size) {
      return false;
    }
  }
  return true;
}

// todo: why return a vector?
std::vector<ParallelTensorShape>
    get_output_shapes(GatherAttrs const &attrs,
                      ParallelTensorShape const &lhs,
                      ParallelTensorShape const &rhs) {
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
