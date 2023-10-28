#include "op-attrs/ops/reduce.h"
#include "utils/exception.decl.h"
#include "utils/exception.h"

namespace FlexFlow {

//
ParallelTensorShape get_output_shape(ReduceAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() - attrs.axes.size() == 1) {
    throw mk_runtime_error(" for reduce, the input and attrs.axes must match");
  }
  ParallelTensorShape output = input;
  for (int i = 0; i < attrs.axes.size(); i++) {
    output.at(attrs.axes.at(i)).size = 1;
    output.at(attrs.axes.at(i)).is_replica_dim = false;
  }

  return output;
}

} // namespace FlexFlow
