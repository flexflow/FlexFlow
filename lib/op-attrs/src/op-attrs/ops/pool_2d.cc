#include "op-attrs/ops/pool_2d.h"

namespace FlexFlow {

TensorShape get_output_shape(Pool2DAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(Pool2DAttrs const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

/*
#include "op-attrs/ops/pool_2d.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"

namespace FlexFlow {

namespace Input {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

namespace Output {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
};

bool Pool2DAttrs::is_valid(ParallelTensorShape const &input) const {
  ParallelTensorShape output_shape = this->calculate_output_shape(input);

  return output_shape.is_valid() && (input.at(Input::REPLICA).degree == 1);
}

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

ParallelTensorShape Pool2DAttrs::calculate_output_shape(ParallelTensorShape
const &input) const { return solve_mappings(input).output_shapes.at(0);
}

} // namespace FlexFlow
*/
