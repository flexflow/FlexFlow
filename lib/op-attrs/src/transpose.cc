#include "op-attrs/ops/transpose.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

// assume we have [x, y, z, l], perms is [0,2] we return [z, y, x, l]
ParallelTensorShape get_output_shape(TransposeAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (attrs.perm.size() != 2) {
    throw mk_runtime_error("TransposeAttrs: perm.size() != 2");
  }

  auto dim0 = attrs.perm[0];
  auto dim1 = attrs.perm[1];
  if (dim0 < 0 || dim1 < 0 || dim0 >= input.num_dims() ||
      dim1 >= input.num_dims()) {
    throw mk_runtime_error("TransposeAttrs: dim0 < 0 || dim1 < 0 || dim0 >= "
                           "input.num_dims() || dim1 >= input.num_dims()");
  }

  ParallelTensorShape output = input;
  int temp = input.at(ff_dim_t(dim0)).size;
  int degree = input.at(ff_dim_t(dim0)).degree;
  output.at(ff_dim_t(dim0)).size = input.at(ff_dim_t(dim1)).size;
  output.at(ff_dim_t(dim1)).size = temp;
  output.at(ff_dim_t(dim0)).degree = input.at(ff_dim_t(dim1)).degree;
  output.at(ff_dim_t(dim1)).degree = degree;
  output.at(ff_dim_t(dim0)).is_replica_dim =
      output.at(ff_dim_t(dim0)).degree > 1;
  output.at(ff_dim_t(dim1)).is_replica_dim =
      output.at(ff_dim_t(dim1)).degree > 1;
  return output;
}

}

} // namespace FlexFlow
