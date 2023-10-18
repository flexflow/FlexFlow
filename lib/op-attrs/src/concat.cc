#include "op-attrs/ops/concat.h"
#include "utils/exception.h"

namespace FlexFlow {
ParallelTensorShape
    get_output_shape(ConcatAttrs const &attrs,
                     std::vector<ParallelTensorShape> const &inputs) {
  ParallelTensorShape output = inputs[0];
  for (auto &i : inputs) {
    if (attrs.axis >= i.num_dims() || i.is_valid() == false) {
      throw mk_runtime_error("ConcatAttrs::get_output_shape: axis is out of "
                             "range or input is invalid");
    }
  }
  for (auto &i : inputs) {
    output.at(attrs.axis).size += i.at(attrs.axis).size;
  }
  output.at(attrs.axis).degree = inputs[0].at(attrs.axis).degree;
  output.at(attrs.axis).is_replica_dim = inputs[0].at(attrs.axis).degree >= 1;
  return output;
}

} // namespace FlexFlow
