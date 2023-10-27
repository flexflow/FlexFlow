#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"
#include "utils/exception.h"
#include "utils/vector.h"

namespace FlexFlow {

namespace Input {
constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4,
              NUMDIM = 5;
}

namespace Output {
constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4,
              NUMDIM = 5;
}

namespace Kernel {
constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL_IN = 2, CHANNEL_OUT = 3,
              REPLICA = 4;
constexpr int WEIGHT_IDX = 0;
} // namespace Kernel

namespace Bias {
constexpr int CHANNEL = 0, REPLICA_1 = 1, REPLICA_2 = 2, REPLICA_3 = 3,
              REPLICA_4 = 4;
constexpr int WEIGHT_IDX = 1;
} // namespace Bias

static std::vector<ParallelDimMappingRecord>
    construct_output_mappings(ParallelTensorShape const &input_shape) {
  return construct_output_parallel_dims(
      {{Input::CHANNEL, MappingOperation::REPLICATE, Output::REPLICA},
       {Input::SAMPLE, MappingOperation::PARTITION, Output::SAMPLE},
       {Input::REPLICA, MappingOperation::PARTITION, Output::CHANNEL},
       {Input::HEIGHT, MappingOperation::PARTITION, Output::HEIGHT},
       {Input::WIDTH, MappingOperation::PARTITION, Output::WIDTH}});
}

static std::vector<ParallelDimMappingRecord>
    construct_kernel_mappings(ParallelTensorShape const &input_shape) {
  return construct_weight_parallel_dims(
      {
          {Input::REPLICA, MappingOperation::PARTITION, Kernel::CHANNEL_OUT},
          {Input::SAMPLE, MappingOperation::REPLICATE, Kernel::REPLICA},
          {Input::CHANNEL, MappingOperation::PARTITION, Kernel::CHANNEL_IN},
          {Input::HEIGHT,
           MappingOperation::REPLICATE,
           Kernel::HEIGHT}, // Kernel::{HEIGHT, WEIGHT} would both work
                            // here
          {Input::WIDTH,
           MappingOperation::REPLICATE,
           Kernel::WIDTH}, // same as above
      },
      0,
      Kernel::WEIGHT_IDX);
}

static std::vector<ParallelDimMappingRecord>
    construct_bias_mappings(ParallelTensorShape const &input_shape) {
  return construct_weight_parallel_dims({{Input::REPLICA, Bias::REPLICA_1},
                                         {Input::SAMPLE, Bias::REPLICA_2},
                                         {Input::CHANNEL, Bias::CHANNEL},
                                         {Input::HEIGHT, Bias::REPLICA_3},
                                         {Input::WIDTH, Bias::REPLICA_4}},
                                        0,
                                        Bias::WEIGHT_IDX);
}

std::vector<ParallelDimMappingRecord>
    construct_mappings(ParallelTensorShape const &input_shape, bool use_bias) {
  std::vector<ParallelDimMappingRecord> mappings =
      concat(construct_output_mappings(input_shape),
             construct_kernel_mappings(input_shape));
  if (use_bias) {
    std::vector<ParallelDimMappingRecord> bias_mappings =
        construct_bias_mappings(input_shape);
    mappings.insert(mappings.end(), bias_mappings.begin(), bias_mappings.end());
  }

  return mappings;
}

// input: (<ri, di1, t>, <b, 1, f>, <input_channel, di3, f>, < input_h, di4, f>,
// <input_w, di5, f> ) kernel(Conv2DAttrs):  out_channels, kernel_h, kernel_w,
// stride_h, stride_w, padding_h, padding_w, output shape:(<ro, do1, t>, <b,
// 1,f>, <output_channel, do3, f>, <output_h, do4, f>, <output_w, do5,f>)
//  output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1
//  output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1
// assert: for the kernel, dk1 == dk2=dk4=dk4=dk5=1
// question:how to decide the ro/do3/do4/do5?
// I think: do3= di3, di4= do4, di5 = do5, do1=di1, ro=ri
ParallelTensorShape get_output_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() != 5) {
    throw mk_runtime_error("Conv2DAttrs::get_output_shape: input is invalid");
  }
  if (attrs.kernel_h > input.at(ff_dim_t(3)).size ||
      attrs.kernel_w > input.at(ff_dim_t(4)).size) {
    throw mk_runtime_error(
        "Conv2DAttrs::get_output_shape: kernel size is larger than input size");
  }

  ParallelTensorShape output = input;
  output.at(ff_dim_t(0)).is_replica_dim = true;
  output.at(ff_dim_t(2)).size = attrs.out_channels;
  output.at(ff_dim_t(3)).size =
      (input.at(ff_dim_t(3)).size + 2 * attrs.padding_h - attrs.kernel_h) /
          attrs.stride_h +
      1;
  output.at(ff_dim_t(4)).size =
      (input.at(ff_dim_t(4)).size + 2 * attrs.padding_w - attrs.kernel_w) /
          attrs.stride_w +
      1;
  for (int i = 1; i < output.num_dims(); i++) {
    output.at(ff_dim_t(i)).is_replica_dim = false;
  }

  return output;
}

} // namespace FlexFlow
