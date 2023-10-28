#include "op-attrs/ops/linear.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// torch.nn.Linear(in_features, out_features, bias=True, device=None,
// dtype=None)
//  pytorch: input shape:{batch_size, input_channels}
//  pytorch linearattrs: should be {input_channels, output_channels}
//  pytorch: output shape:{batch_size, output_channels}
//  question: the Linearattrs doesn't have input_channels
// input: (<ri, di1, t>, <b, di2, f>, <input_channels, di3, f>)
// linearattrs: should be {input_channels, output_channels}
// the Linearattrs doesn't have input_channels, just have output_channels
// output:(<ro,do1, t>, <b, do2, f>, <output_channels, do3, f>>
// I think do1 = di1, do = ri, do2= di2, do3 = di3

ParallelTensorShape get_output_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output_shape = input;
  if (input.num_dims() != 3) {
    throw mk_runtime_error("LinearAttrs: input shape should be 3D");
  }

  output_shape.at(ff_dim_t(2)).size = attrs.out_channels;
  return output_shape;
}

} // namespace FlexFlow
