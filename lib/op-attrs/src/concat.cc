#include "op-attrs/ops/concat.h"
#include "utils/exception.decl.h"
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

  int dims = inputs[0].num_dims();
  for (int i = 1; i < inputs.size(); i++) {
    if (inputs[i].num_dims() != dims) {
      throw mk_runtime_error(" the input dims not matched at i:", i);
    }
  }

  for (auto &i : inputs) {
    output.at(ff_dim_t(attrs.axis)).size += i.at(ff_dim_t(attrs.axis)).size;
  }
  output.at(ff_dim_t(0)).is_replica_dim = true;
  // note: how to decide the degee?
  for (int i = 1; i < output.num_dims(); i++) {
    output.at(ff_dim_t(i)).is_replica_dim = false;
  }
  return output;
}

} // namespace FlexFlow
