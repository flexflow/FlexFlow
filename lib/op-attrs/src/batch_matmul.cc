#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.h"

namespace FlexFlow {

// how to get the batch size? and lhs: [b, n, m], rhs: [b, m, p]
// output: [b, n, p] //n == s1, m == s2
//[b, n/2, m], [b, m, p/2] -> [b, n/2, p/2]
//[b, n, m/2], [b, m/2, p] -> [b, n, p/2]
ParallelTensorShape get_output_shape(BatchMatmulAttrs const &attrs,
                                     ParallelTensorShape const &lhs,
                                     ParallelTensorShape const &rhs) {
  ParallelTensorShape output_shape = lhs;

  // check if the input is valid
  if (!lhs.is_valid() || !rhs.is_valid()) {
    throw mk_runtime_error(
        "BatchMatmulAttrs::get_output_shape: input is invalid")
  }

  if (lhs.at(ff_dim_t(0)).size != rhs.at(ff_dim_t(0)).size) {
    throw mk_runtime_error(
        "BatchMatmulAttrs::get_output_shape: batch size is not equal");
  }
  if (lhs.at(ff_dim_t(2)).size != rhs.at(ff_dim_t(1)).size ||
      lhs.at(ff_dim_t(1)).size != attrs.a_seq_length_dim ||
      rhs.at(ff_dim_t(2)).size != attrs.b_seq_length_dim) {
    throw mk_runtime_error(
        "BatchMatmulAttrs::get_output_shape: third demension of lhs and second "
        "dementions of rhs are not match");
  }
  output_shape.at(ff_dim_t(0)).size = lhs.at(ff_dim_t(0)).size; // batch size
  output_shape.at(ff_dim_t(1)).size = lhs.at(ff_dim_t(1)).size;
  output_shape.at(ff_dim_t(2)).size = rhs.at(ff_dim_t(2)).size;

  if (lhs.at(ff_dim_t(1)).degree == 1 && lhs.at(ff_dim_t(2)).degree == 1) {
    // case 0: degree is 1, [b, n, m], rhs: [b, m, p] -> [b, n, p]
    for (int i = 1; i < lhs.num_dims(); i++) {
      output_shape.at(ff_dim_t(i)).degree = 1;
      output_shape.at(ff_dim_t(i)).is_replica_dim = false;
    }
  } else if (lhs.at(ff_dim_t(1)).degree == 1 &&
             lhs.at(ff_dim_t(2)).degree >
                 1) { // case 1: [b, n, m/x], [b, m/x, p] => [b, n, y]
    output_shape.at(ff_dim_t(1)).is_replica_dim = true;
    output_shape.at(ff_dim_t(1)).degree = lhs.at(ff_dim_t(1)).degree;
  } else if (lhs.at(ff_dim_t(1)).degree > 1 &&
             lhs.at(ff_dim_t(2)).degree ==
                 1) { // case 2: [b, n/x, m] [b m p/x] => [b n/x p/x]
    output_shape.at(ff_dim_t(1)).is_replica_dim = true;
    output_shape.at(ff_dim_t(2)).is_replica_dim = true;
    output_shape.at(ff_dim_t(1)).degree = lhs.at(ff_dim_t(1)).degree;
    output_shape.at(ff_dim_t(2)).degree = rhs.at(ff_dim_t(2)).degree;
  } else if (lhs.at(ff_dim_t(1)).degree > 1 &&
             lhs.at(ff_dim_t(2)).degree >
                 1) { // case 3: [b n/x m/y] [b m/y p/x]=> [b n/x p/x]
    output_shape.at(ff_dim_t(1)).is_replica_dim = true;
    output_shape.at(ff_dim_t(2)).is_replica_dim = true;
    output_shape.at(ff_dim_t(1)).degree = lhs.at(ff_dim_t(1)).degree;
    output_shape.at(ff_dim_t(2)).degree = rhs.at(ff_dim_t(2)).degree;
  } else {
    throw mk_runtime_error("BatchMatmulAttrs::get_output_shape: not supported "
                           "in BatchMatmulAttrs get_output_shape");
  }
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
