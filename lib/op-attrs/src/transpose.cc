#include "op-attrs/ops/transpose.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool TransposeAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  // in pytorch, we use choose two dim for transpose, so I think the size of
  // perm should be 2
  if (perm.size() != 2) {
    return false;
  }

  auto dim0 = perm[0];
  auto dim1 = perm[1];
  if (dim0 < 0 || dim1 < 0 || dim0 >= input.num_dims() ||
      dim1 >= input.num_dims()) {
    return false;
  }

  return true;
}

// assume we have [x, y, z, l], perms is [0,2] we return [z, y, x, l]
ParallelTensorShape get_output_shape(TransposeAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  auto dim0 = attrs.perm[0];
  auto dim1 = attrs.perm[1];
  int temp = input.at(ff_dim_t(dim0)).size;
  output.at(ff_dim_t(dim0)).size = input.at(ff_dim_t(dim1)).size;
  output.at(ff_dim_t(dim1)).size = temp;
  return output;
}

}

} // namespace FlexFlow
