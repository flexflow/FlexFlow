#include "op-attrs/ops/batch_matmul.h"

namespace FlexFlow {

 bool BatchMatmulAttrs::is_valid(ParallelTensorShape const & lhs, ParallelTensorShape const & rhs) {
    if (!lhs.is_valid() || !rhs.is_valid()) {
          return false;
    }
    if (lhs.num_dims() != rhs.num_dims()) {
          return false;
    }
    return true;
}

ParallelTensorShape get_output_shape(BatchMatmulAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &) {

}


/* bool BatchMatmulAttrs::is_valid( */
/*     ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
 */
/*   if (!lhs.is_valid() || !rhs.is_valid()) { */
/*     return false; */
/*   } */
/*   if (lhs.num_dims() != rhs.num_dims()) { */
/*     return false; */
/*   } */
/*   for (int i = lhs.num_dims() - 1; i >= 2; i--) { */
/*     if (lhs.at(i) != rhs.at(i)) { */
/*       return false; */
/*     } */
/*   } */
/*   if (lhs.at(0) != rhs.at(1)) { */
/*     return false; */
/*   } */

/*   return true; */
/* } */

} // namespace FlexFlow
