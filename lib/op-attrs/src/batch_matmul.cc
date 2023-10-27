#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.decl.h"
#include "utils/exception.h"

namespace FlexFlow {

// lhs: [<r1, dl1, true>, <b, 1, f> ,<n, dl3, false>, <m, dl4, false>]
// rhs:[<r2, dr1, true>, <b,1,f> ,<m, dr3, false>, <p, dr4,false>]
// in the original tensor, we assume the dl1/dr1 is 1
// output:[<r3, do1, true>, <b,1,f>, <n, do3, false>, <p,do4, false>]
// how to decide the r3, d01, do3, do4
// Note: Lsize = r1 * dl3 * dl4, Rsize = r2 * dr3 * dr4 , Rsize = Lsize
// do3 = dl3, do4 = dr4
// so, r3 = Lsize / do3 / do4
// r3 / do1 = r1 / dl1
ParallelTensorShape get_output_shape(BatchMatmulAttrs const &attrs,
                                     ParallelTensorShape const &lhs,
                                     ParallelTensorShape const &rhs) {
  if (lhs.num_dims() != 4 || rhs.num_dims() != 4) {
    throw mk_runtime_error("rhs or lhs dimension is not 4");
  }

  int rl = lhs.at(ff_dim_t(0)).size;    // replicate_num of lhs
  int dl1 = lhs.at(ff_dim_t(0)).degree; // degree of 0 dimension
  int dl3 = lhs.at(ff_dim_t(3)).degree; // degree of third dimension
  int dr4 = rhs.at(ff_dim_t(4)).degree; // degree of fouth dimenstion

  int lsize = lhs.get_volume();
  int rsize = rhs.get_volume();
  if (lsize != rsize) {
    throw mk_runtime_error("BatchMatmulAttrs::get_output_shape, the volume of "
                           "lhs and rhs are not matched ");
  }

  if (lhs.at(ff_dim_t(1)).size != rhs.at(ff_dim_t(1)).size) {
    throw mk_runtime_error(
        "BatchMatmulAttrs::get_output_shape, batch size is not equal");
  }

  if (lhs.at(ff_dim_t(3)).size != rhs.at(ff_dim_t(3)).size) {
    throw mk_runtime_error(
        "BatchMatmulAttrs::get_output_shape: forth demension of lhs and third "
        "dementions of rhs are not match");
  }

  // 4D tensor
  ParallelTensorShape output_shape = lhs;

  output_shape.at(ff_dim_t(0)).size = lsize / (dl3 * dr4);
  output_shape.at(ff_dim_t(0)).degree =
      output_shape.at(ff_dim_t(0)).size /
      (rl / dl1); // this may have some problem
  output_shape.at(ff_dim_t(0)).is_replica_dim = true;

  output_shape.at(ff_dim_t(3)).size = lhs.at(ff_dim_t(3)).size;
  output_shape.at(ff_dim_t(3)).degree = dl3;
  output_shape.at(ff_dim_t(3)).is_replica_dim = false;

  output_shape.at(ff_dim_t(4)).size = rhs.at(ff_dim_t(4)).size();
  output_shape.at(ff_dim_t(4)).degree = dr4;
  output_shape.at(ff_dim_t(4)).is_replica_dim = false;

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
