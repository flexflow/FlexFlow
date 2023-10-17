#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ff_dim.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"
#include "utils/exception.h"

namespace FlexFlow {

namespace Input {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

namespace Output {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

/* bool Pool2DAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   ParallelTensorShape output_shape = this->calculate_output_shape(input); */

/*   return output_shape.is_valid() && (input.at(Input::REPLICA).degree == 1);
 */
/* } */

static std::vector<ParallelDimMappingRecord>
    construct_mappings(ParallelTensorShape const &input_shape) {
  auto const outputMappings = construct_output_parallel_dims({
      {Input::REPLICA, MappingOperation::PARTITION, Output::REPLICA},
      {Input::SAMPLE, MappingOperation::PARTITION, Output::SAMPLE},
      {Input::CHANNEL, MappingOperation::PARTITION, Output::CHANNEL},
      {Input::HEIGHT, MappingOperation::PARTITION, Output::HEIGHT},
      {Input::WIDTH, MappingOperation::PARTITION, Output::WIDTH},
  });

  return outputMappings;
}

static ParallelDimMappingSolution
    solve_mappings(ParallelTensorShape const &input) {
  return solve_parallel_dim_mappings(construct_mappings(input), {input}, 0, 1);
}

bool Pool2DAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  return true;
}

// https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
//  pytorch: we have two type of pool2d, maxpool2d and avgpool2d
//  input shape: (batch_size, channels, input_height, input_width)
//  for avgpool2d, output shape: (batch_size, channels, 1, 1)
//  for maxpool2d, output shape: (batch_size, channels, output_height,
//  output_width) output_height = (input_height + 2 * padding_h - kernel_h) /
//  stride_h + 1 output_width = (input_width + 2 * padding_w - kernel_w) /
//  stride_w + 1
ParallelTensorShape get_output_shape(Pool2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {

  if (input.num_dims() != 4) {
    throw mk_runtime_error("Pool2DAttrs: input shape should be 4D");
  }
  ParallelTensorShape output_shape = input;
  if (attrs.pool_type == PoolOp::AVG) {
    output_shape.at(ff_dim_t(2)).size = 1;
    output_shape.at(ff_dim_t(3)).size = 1;
  } else if (attrs.pool_type == PoolOp::MAX) {
    output_shape.at(ff_dim_t(2)).size =
        (input.at(ff_dim_t(2)).size + 2 * attrs.padding_h - attrs.kernel_h) /
            attrs.stride_h +
        1;
    output_shape.at(ff_dim_t(3)).size =
        (input.at(ff_dim_t(3)).size + 2 * attrs.padding_w - attrs.kernel_w) /
            attrs.stride_w +
        1;
  } else {
    throw mk_runtime_error("Pool2DAttrs: pool type is not supported");
  }

  // case 1: input:[N, C, H, W], output:[N, C, 1, 1], degree is 1 for avgpool2d
  // input: [N, C, H, W], output: [N, C, output_height, output_width], degree is  1 for maxpool2d
  if (input.at(ff_dim_t(2)).degree == 1 && input.at(ff_dim_t(3)).degree == 1) {
    for (int i = 2; i < input.num_dims(); i++) {
      output_shape.at(ff_dim_t(i)).is_replica_dim = false;
      output_shape.at(ff_dim_t(i)).degree = 1;
    }
  } else if (input.at(ff_dim_t(2)).degree > 1 &&
             input.at(ff_dim_t(3)).degree == 1) {
    // case 2: input [N, C, H/X, W] output [N, C, 1, 1], degree is X
    // input [N, C, H/X, W] output [N, C, output_height/x, output_width], degree  is X
    output_shape.at(ff_dim_t(2)).degree = input.at(ff_dim_t(2)).degree;
    output_shape.at(ff_dim_t(2)).is_replica_dim = true;
    output_shape.at(ff_dim_t(3)).degree = 1;
    output_shape.at(ff_dim_t(3)).is_replica_dim = false;
  } else if (input.at(ff_dim_t(2)).degree == 1 &&
             input.at(ff_dim_t(3)).degree > 1) {
    // case 3: input [N, C, H, W/X] output [N, C, 1, 1], degree is X
    // input [N, C, H, W/X] output [N, C, output_height, output_width/x], degree is X
    output_shape.at(ff_dim_t(2)).degree = 1;
    output_shape.at(ff_dim_t(2)).is_replica_dim = false;
    output_shape.at(ff_dim_t(3)).degree = input.at(ff_dim_t(3)).degree;
    output_shape.at(ff_dim_t(3)).is_replica_dim = true;
  } else if (input.at(ff_dim_t(2)).degree > 1 &&
             input.at(ff_dim_t(3)).degree > 1) {
    // case 4: input [N, C, H/X, W/Y] output [N, C, 1, 1], degree is X and Y for
    // avgpool2d input [N, C, H/X, W/Y] output [N, C, output_height/x, output_width/y], degree is X and Y for maxpool2d
    for (int i = 2; i < input.num_dims(); i++) {
      output_shape.at(ff_dim_t(i)).is_replica_dim = true;
      output_shape.at(ff_dim_t(i)).degree = input.at(ff_dim_t(i)).degree;
    }
  } else {
    throw mk_runtime_error("Pool2DAttrs: degree is not supported");
  }

  return output_shape;
}

}

/* ParallelTensorShape Pool2DAttrs::calculate_output_shape(ParallelTensorShape
 * const &input) const { */
/*   return solve_mappings(input).output_shapes.at(0); */
/* } */

} // namespace FlexFlow
