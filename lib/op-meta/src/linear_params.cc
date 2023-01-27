#include "op-meta/ops/linear_params.h"
#include "utils/hash-utils.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"

namespace FlexFlow {

namespace Input {
  constexpr int CHANNEL = 0, SAMPLES = 1, REPLICA = 2, NUMDIM = 3;
}

namespace Output {
  constexpr int CHANNEL = 0, SAMPLES = 1, REPLICA = 2, NUMDIM = 3;
}

namespace Kernel {
  constexpr int CHANNEL_IN = 0, CHANNEL_OUT = 1, NUMDIM = 2;
  constexpr int WEIGHT_IDX = 0;
};

namespace Bias {
  constexpr int CHANNEL_OUT = 0, NUMDIM = 1;
  constexpr int WEIGHT_IDX = 1;
};

bool LinearParams::is_valid(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output_shape = this->calculate_output_shape(input_shape);
  ParallelTensorShape kernel_shape = this->calculate_kernel_shape(input_shape);

  bool is_valid = true;
  is_valid &= input_shape.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= kernel_shape.is_valid();
  if (use_bias) {
    ParallelTensorShape bias_shape = this->calculate_bias_shape(input_shape);
    is_valid &= bias_shape.is_valid();
  }
  return is_valid;
};

std::vector<ParallelDimMappingRecord> construct_mappings(
    ParallelTensorShape const &input_shape) {

  auto const outputMappings = [&]() -> std::vector<ParallelDimMappingRecord> {
    std::vector<ParallelDimMappingRecord> m = construct_output_parallel_dims(
        {{Input::CHANNEL, Output::REPLICA},
         {Input::REPLICA, Output::CHANNEL}});
    
    for (int i = 1; i < input_shape.num_dims() - 1; i++) {
      m.push_back(construct_output_parallel_dims(i, i));
    }

    return m;
  }();

  auto const kernelMappings = [&]() -> std::vector<ParallelDimMappingRecord> {
    std::vector<ParallelDimMappingRecord> m = construct_weight_parallel_dims(
                                       {{Input::CHANNEL,
          Kernel::CHANNEL_IN},
                                        {Input::REPLICA,
            Kernel::CHANNEL_OUT}},
                                       0 /*input_idx*/,
                                       Kernel::WEIGHT_IDX);
    // map a bunch of replica dimensions for the unnamed dimensions in the input
    for (int i = 1; i < input_shape.num_dims() - 1; i++) {
      m.push_back(construct_weight_parallel_dims(i, i + 1, 0 /*input_idx*/, Kernel::WEIGHT_IDX));
    }

    return m;
  }();

  auto const biasMappings = [&]() -> std::vector<ParallelDimMappingRecord> {
    std::vector<ParallelDimMappingRecord> m = construct_weight_parallel_dims(
                                     { {Input::REPLICA, Bias::CHANNEL_OUT}, },
                                     0 /*input_idx*/,
                                     Bias::WEIGHT_IDX);
    for (int i = 0; i < input_shape.num_dims() - 1; i++) {
      m.push_back(construct_weight_parallel_dims(i, i + 1, 0 /*input_idx*/, Bias::WEIGHT_IDX));
    }
    return m;
  }();

  std::vector<ParallelDimMappingRecord> allMappings;
  allMappings.insert(allMappings.end(), outputMappings.begin(), outputMappings.end());
  allMappings.insert(allMappings.end(), kernelMappings.begin(), kernelMappings.end());
  allMappings.insert(allMappings.end(), biasMappings.begin(), biasMappings.end());

  return allMappings;
}

ParallelDimMappingSolution solve_mappings(ParallelTensorShape const &input) {
  return solve_parallel_dim_mappings(construct_mappings(input), {input}, 2, 1);
}

ParallelTensorShape LinearParams::calculate_output_shape(ParallelTensorShape const &input) const {
  return solve_mappings(input).output_shapes.at(0);
}

ParallelTensorShape LinearParams::calculate_kernel_shape(ParallelTensorShape const &input) const {
  return solve_mappings(input).weight_shapes.at(Kernel::WEIGHT_IDX);
}

typename LinearParams::AsConstTuple LinearParams::as_tuple() const {
  return {this->out_channels, this->use_bias, this->data_type, this->activation};
}

bool operator==(LinearParams const &lhs, LinearParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(LinearParams const &lhs, LinearParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::LinearParams>::operator()(
    FlexFlow::LinearParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
