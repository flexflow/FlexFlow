#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(Pool2DAttrs const &attrs,
                             TensorShape const &input_shape) {
  size_t num_samples = dim_at_idx(input_shape, ff_dim_t{0});
  size_t num_channels = dim_at_idx(input_shape, ff_dim_t{1});
  size_t input_height = dim_at_idx(input_shape, ff_dim_t{2});
  size_t input_width = dim_at_idx(input_shape, ff_dim_t{3});

  size_t output_height =
      (input_height + 2 * attrs.padding_h - attrs.kernel_h) / attrs.stride_h +
      1;

  size_t output_width =
      (input_width + 2 * attrs.padding_w - attrs.kernel_w) / attrs.stride_w + 1;

  return TensorShape{TensorDims{FFOrdered<size_t>{
                         num_samples,
                         num_channels,
                         output_height,
                         output_width,
                     }},
                     input_shape.data_type};
}
// TODO(@pietro): add tests for this and concat

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
