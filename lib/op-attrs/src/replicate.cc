#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_dim.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool ReplicateAttrs::is_valid(ParallelTensorShape const &input) const {
  if(!input.is_valid()) {
    return false;
  }
  if(this->replicate_dim >= input.num_dims() || this->replicate_degree <= 0) {
    return false;
  }

  return true;
}

//replicate by n multiplies degree by n and shape by n
//seems it is like pytorch's repeat
//original_tensor = torch.tensor([1, 2, 3]) torch.Size([3])
///replicated_tensor = original_tensor.repeat(3) torch.Size([9])

//original_tensor = torch.randn(2, 3, 4) torch.Size([2, 3, 4])
//repeated_tensor = original_tensor.repeat(3, 1, 1) torch.Size([6, 3, 4])

ParallelTensorShape get_output_shape(ReplicateAttrs const & attrs,
                                     ParallelTensorShape const & input) {
  assert(attrs.is_valid(input));
  ParallelTensorShape output = input;
  output.at(attrs.replicate_dim).size *= attrs.replicate_degree;
  return output;
}



} // namespace FlexFlow
