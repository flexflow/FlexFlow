#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

//maybe we should add more check here
bool BatchMatmulAttrs::is_valid(ParallelTensorShape const & lhs, ParallelTensorShape const & rhs) {
    if (!lhs.is_valid() || !rhs.is_valid()) {
          return false;
    }
    if (lhs.num_dims() != rhs.num_dims()) {
          return false;
    }
    return true;
}

//how to get the batch size? and lhs: [b, s1, k], rhs: [b, k, s1]
ParallelTensorShape get_output_shape(BatchMatmulAttrs const & attrs,
                                     ParallelTensorShape const & lhs,
                                     ParallelTensorShape const & rhs) {
  ParallelTensorShape   output_shape = lhs;
  output_shape.at(ff_dim_t(0)).size = lhs.at(ff_dim_t(0)).size;
  output_shape.at(ff_dim_t(1)).size = attrs.a_seq_length_dim;
  output_shape.at(ff_dim_t(2)).size = attrs.b_seq_length_dim;
  //TODO: Do we need to set the ParallelDim for output_shape
  return output_shape;  
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
