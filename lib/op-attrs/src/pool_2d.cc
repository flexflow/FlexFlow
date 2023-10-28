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

// https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
// input:(< ri, di1, t>, <b, di2, f>, <channels, di3, f>, <input_height, di4,
// f>, <input_width, di5, f>)

// Pool2DAttrs: req<int> kernel_h, kernel_w, stride_h, stride_w, padding_h,
// padding_w;

// for avgpool2d: output shape:(< ri, di1, t>, <b, di2, f>, <channels, di3, f>,
// <1,1,f>, <1,1,f> )

// for maxpool2d, output shape:(< ri, di1, t>, <b, di2, f>, <channels, di3, f>,
// <output_height, di4, f>, <output_width, di5, f>)

// output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1
// output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1
ParallelTensorShape get_output_shape(Pool2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() != 5) {
    throw mk_runtime_error("Pool2DAttrs, input shape should be 5D");
  }

  if (attrs.pool_type == PoolOp::AVG) {
    std::vector<ParallelDim> data;
    data.resize(4);
    data[0] = input.at(ff_dim_t(0));
    data[1] = input.at(ff_dim_t(1));
    data[2] = {1, 1, false};
    data[3] = {1, 1, false};
    ParallelTensorShape output = ParallelTensorShape(
        ParallelTensorDims(TensorDims(data.begin(), data.end())),
        input.data_type);
    return output;
  } else if (attrs.pool_type == PoolOp::MAX) {
    ParallelTensorShape output_shape = input;
    output_shape.at(ff_dim_t(3)).size =
        (input.at(ff_dim_t(3)).size + 2 * attrs.padding_h - attrs.kernel_h) /
            attrs.stride_h +
        1;
    output_shape.at(ff_dim_t(4)).size =
        (input.at(ff_dim_t(4)).size + 2 * attrs.padding_w - attrs.kernel_w) /
            attrs.stride_w +
        1;
    return output_shape;
  } else {
    throw mk_runtime_error("Pool2DAttrs: pool type is not supported");
  }
}

/* ParallelTensorShape Pool2DAttrs::calculate_output_shape(ParallelTensorShape
 * const &input) const { */
/*   return solve_mappings(input).output_shapes.at(0); */
/* } */

} // namespace FlexFlow
