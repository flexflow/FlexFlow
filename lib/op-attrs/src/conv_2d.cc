#include "op-attrs/ops/conv_2d.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"
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

/* bool Conv2DAttrs::is_valid(ParallelTensorShape const &input_shape) const { */
/*   bool is_valid = true; */
/*   is_valid &= input_shape.is_valid(); */
/*   is_valid &= this->calculate_output_shape(input_shape).is_valid(); */
/*   is_valid &= this->calculate_kernel_shape(input_shape).is_valid(); */
/*   if (use_bias) { */
/*     is_valid &= this->calculate_bias_shape(input_shape).is_valid(); */
/*   } */

/*   // TODO FIXME: Currently disable parallelizing the height and width
 * dimension */
/*   if (input_shape.at(0).degree > 1 || input_shape.at(1).degree > 1) { */
/*     return false; */
/*   } */

/*   return is_valid; */

/* } */

/* OperatorType Conv2DAttrs::op_type() const { */
/*   return OP_CONV2D; */
/* } */

} // namespace FlexFlow
