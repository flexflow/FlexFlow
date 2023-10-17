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
ParallelTensorShape get_output_shape(LinearAttrs const &atts,
                                     ParallelTensorShape const &input) {

  ParallelTensorShape out_shape = input;
  if (input.num_dims() != 2) {
    throw mk_runtime_error("LinearAttrs: input shape should be 2D");
  }

  out_shape.at(ff_dim_t(1)).size = atts.out_channels;
  // linear shoud consider the degree
  // case 1: input:[N, K], weight:[K, M], degree is 1
  if (input.at(ff_dim_t(0)).degree == 1 && input.at(ff_dim_t(1)).degree == 1) {
    out_shape.at(ff_dim_t(0)).degree = 1;
    for (int i = 0; i < input.num_dims(); i++) {
      out_shape.at(ff_dim_t(i)).is_replica_dim = false;
      out_shape.at(ff_dim_t(i)).degree = 1;
    }
  } else if (input.at(ff_dim_t(0)).degree == 1 &&
             input.at(ff_dim_t(1)).degree > 1) {
    // case 2: input [N, k/x], weight [k/x, M], output [N, M], degree is x
    out_shape.at(ff_dim_t(1)).degree = input.at(ff_dim_t(1)).degree;
    out_shape.at(ff_dim_t(1)).is_replica_dim = true;
  } else if (input.at(ff_dim_t(0)).degree > 1 &&
             input.at(ff_dim_t(1)).degree == 1) {
    // case 3: input [N/X, K], weight [K, M/X], output [N/X, M], degree is X
    out_shape.at(ff_dim_t(0)).degree = input.at(ff_dim_t(0)).degree;
    out_shape.at(ff_dim_t(0)).is_replica_dim = true;
  } else if (input.at(ff_dim_t(0)).degree > 1 &&
             input.at(ff_dim_t(1)).degree > 1) {
    // case 4: input [N/X, K/Y], weight [K/Y, M/X], output [N/X, M/X], degree is
    // X
    for (int i = 0; i < input.num_dims(); i++) {
      out_shape.at(ff_dim_t(i)).is_replica_dim = true;
      out_shape.at(ff_dim_t(i)).degree = input.at(ff_dim_t(i)).degree;
    }
  } else {
    throw mk_runtime_error("LinearAttrs: degree is not supported");
  }
  return out_shape;
}

} // namespace FlexFlow
