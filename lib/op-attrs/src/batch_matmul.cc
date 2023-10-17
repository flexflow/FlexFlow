#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

// maybe we should add more check here
//// how to get the batch size? and lhs: [b, n, m], rhs: [b, m, p]
// output: [b, n, p] //n == s1, m == s2
//[n/]
bool is_valid(BatchMatmulAttrs const &attrs,
              ParallelTensorShape const &lhs,
              ParallelTensorShape const &rhs) {
  if (lhs.at(ff_dim_t(0)).size != rhs.at(ff_dim_t(0)).size) {
    return false;
  }
  if (lhs.at(ff_dim_t(2)).size != rhs.at(ff_dim_t(1)).size) {
    return false;
  }
  if (lhs.at(ff_dim_t(1)).size != attrs.a_seq_length_dim) {
    return false;
  }

  if (rhs.at(ff_dim_t(2)).size != attrs.b_seq_length_dim) {
    return false;
  }

  return true;
}

// how to get the batch size? and lhs: [b, n, m], rhs: [b, m, p]
// output: [b, n, p] //n == s1, m == s2
//[b, n/2, m], [b, m, p/2] -> [b, n/2, p/2]
//[b, n, m/2], [b, m/2, p] -> [b, n, p/2]
ParallelTensorShape get_output_shape(BatchMatmulAttrs const &attrs,
                                     ParallelTensorShape const &lhs,
                                     ParallelTensorShape const &rhs) {
  ParallelTensorShape output_shape = lhs;
  output_shape.at(ff_dim_t(0)).size = lhs.at(ff_dim_t(0)).size;
  // degree is 1
  //[b, n, m], rhs: [b, m, p] -> [b, n, p]
  if (lhs.at(ff_dim_t(1)).degree == 1 && rhs.at(ff_dim_t(2)).degree == 1) {
    output_shape.at(ff_dim_t(1)).size = lhs.at(ff_dim_t(1)).size;
    output_shape.at(ff_dim_t(2)).size = rhs.at(ff_dim_t(2)).size;
    output_shape.at(ff_dim_t(0)).is_replica_dim = false;
  } else if (lhs.at(ff_dim_t(1)).degree > 1 &&
             rhs.at(ff_dim_t(2)).degree ==
                 1) { //[b, n/x, m], [b, m, p/x] => [b, n/x, p/x]
    output_shape.at(ff_dim_t(1)).size =
        lhs.at(ff_dim_t(1)).size / lhs.at(ff_dim_t(1)).degree;
    output_shape.at(ff_dim_t(2)).size =
        rhs.at(ff_dim_t(2)).size / rhs.at(ff_dim_t(2)).degree;
    output_shape.at(ff_dim_t(0)).is_replica_dim = true;
  } else if (lhs.at(ff_dim_t(1)).degree == 1 &&
             rhs.at(ff_dim_t(2)).degree >
                 1) { //[b, n, m/x], [b, m/x, p] => [b, n, p/x]
    output_shape.at(ff_dim_t(1)).size = lhs.at(ff_dim_t(1)).size;
    output_shape.at(ff_dim_t(2)).size =
        rhs.at(ff_dim_t(2)).size / rhs.at(ff_dim_t(2)).degree;
    output_shape.at(ff_dim_t(0)).is_replica_dim = true;
  } else if (lhs.at(ff_dim_t(1)).degree > 1 &&
             rhs.at(ff_dim_t(2)).degree >
                 1) { //[b, n/x, m/y], [b, m/y, p/z] => [b, n/x, p/z]
    output_shape.at(ff_dim_t(1)).size =
        lhs.at(ff_dim_t(1)).size / lhs.at(ff_dim_t(1)).degree;
    output_shape.at(ff_dim_t(2)).size =
        rhs.at(ff_dim_t(2)).size / rhs.at(ff_dim_t(2)).degree;
    output_shape.at(ff_dim_t(0)).is_replica_dim = true;
  } else {
    assert(false && "not supported in BatchMatmulAttrs get_output_shape");
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
