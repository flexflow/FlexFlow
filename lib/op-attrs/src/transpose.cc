#include "op-attrs/ops/transpose.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

// assume input:[<ri, di1, t>, <x,di2, f> , <y, di3, f>, <z, di4, f>]
// perem is [1,2]
// output:[<ri, di1, t>,  <y, di3, f>, <x,di2, f>, <z, di4, f> ]
ParallelTensorShape get_output_shape(TransposeAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (attrs.perm.size() != 2) {
    throw mk_runtime_error("TransposeAttrs: perm.size() != 2");
  }

  auto dim0 = attrs.perm[0]; // dim0 and dim1 should not be 0
  auto dim1 = attrs.perm[1];
  if (dim0 <= 0 || dim1 <= 0 || dim0 >= input.num_dims() ||
      dim1 >= input.num_dims()) {
    throw mk_runtime_error("TransposeAttrs: dim0 <= 0 || dim1 <= 0 || dim0 >= "
                           "input.num_dims() || dim1 >= input.num_dims()");
  }

  ParallelTensorShape output = input;
  int temp = input.at(ff_dim_t(dim0)).size;
  int degree = input.at(ff_dim_t(dim0)).degree;
  output.at(ff_dim_t(dim0)).size = input.at(ff_dim_t(dim1)).size;
  output.at(ff_dim_t(dim1)).size = temp;
  output.at(ff_dim_t(dim0)).degree = input.at(ff_dim_t(dim1)).degree;
  output.at(ff_dim_t(dim1)).degree = degree;
  output.at(ff_dim_t(dim0)).is_replica_dim = dim0 == 0;
  output.at(ff_dim_t(dim1)).is_replica_dim = dim1 == 0;
  return output;
}

}

} // namespace FlexFlow
