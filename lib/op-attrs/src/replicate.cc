#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

// replicate by n multiplies degree by n and shape by n
// seems it is like pytorch's repeat
// original_tensor = torch.tensor([1, 2, 3]) torch.Size([3])
/// replicated_tensor = original_tensor.repeat(3) torch.Size([9])

// original_tensor = torch.randn(2, 3, 4) torch.Size([2, 3, 4])
// repeated_tensor = original_tensor.repeat(3, 1, 1) torch.Size([6, 3, 4])

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (attrs.replicate_dim >= input.num_dims() || attrs.replicate_degree <= 0) {
    throw mk_runtime_error("ReplicateAttrs::get_output_shape: axis is out of "
                           "range or input is invalid");
  }
  ParallelTensorShape output = input;
  output.at(attrs.replicate_dim).size *= attrs.replicate_degree;
  return output;
}

} // namespace FlexFlow
