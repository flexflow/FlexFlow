#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ff_dim.h"
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

bool Pool2DAttrs::is_valid(ParallelTensorShape const & input) const {
    if(!input.is_valid()) {
        return false;
    }
    return true;
}

//pytorch: we have two type of pool2d, maxpool2d and avgpool2d
//input shape: (batch_size, channels, input_height, input_width)
//for avgpool2d, output shape: (batch_size, channels, 1, 1)
//for maxpool2d, output shape: (batch_size, channels, output_height, output_width)
//output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1
//output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1
ParallelTensorShape get_output_shape(Pool2DAttrs const & attrs,
                                     ParallelTensorShape const & input) {
    ParallelTensorShape output_shape = input;    
    if(attrs.pool_type == PoolOp::AVG) {
      output_shape.at(ff_dim_t(2)).size = 1;
      output_shape.at(ff_dim_t(3)).size = 1;
    } else if(attrs.pool_type == PoolOp::MAX) {
      output_shape.at(ff_dim_t(2)).size = (input.at(ff_dim_t(2)).size + 2 * attrs.padding_h - attrs.kernel_h) / attrs.stride_h + 1;
      output_shape.at(ff_dim_t(3)).size = (input.at(ff_dim_t(3)).size + 2 * attrs.padding_w - attrs.kernel_w) / attrs.stride_w + 1;
    } else {
      assert(false && "unsupported pool type");
    }
    return output_shape;                                
}

}

/* ParallelTensorShape Pool2DAttrs::calculate_output_shape(ParallelTensorShape
 * const &input) const { */
/*   return solve_mappings(input).output_shapes.at(0); */
/* } */

} // namespace FlexFlow
