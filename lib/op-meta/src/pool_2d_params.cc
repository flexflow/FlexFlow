#include "op-meta/ops/pool_2d_params.h"
#include "utils/hash-utils.h"
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


bool Pool2DParams::is_valid(ParallelTensorShape const &input) const {
  ParallelTensorShape output_shape = this->calculate_output_shape(input);

  bool is_valid = true;
  is_valid &= input.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= (input.at(Input::REPLICA).degree == 1);

  return is_valid;
}

static std::vector<ParallelDimMappingRecord> construct_mappings(ParallelTensorShape const &input_shape) {
  auto const outputMappings = construct_output_parallel_dims( { {Input::REPLICA,
                                          MappingOperation::PARTITION,
                                          Output::REPLICA},
                                         {Input::SAMPLE,
                                          MappingOperation::PARTITION,
                                          Output::SAMPLE},
                                         {Input::CHANNEL,
                                          MappingOperation::PARTITION,
                                          Output::CHANNEL},
                                         {Input::HEIGHT,
                                          MappingOperation::PARTITION,
                                          Output::HEIGHT},
                                         {Input::WIDTH,
                                          MappingOperation::PARTITION,
                                          Output::WIDTH},
                                     });

  return outputMappings;
}

static ParallelDimMappingSolution solve_mappings(ParallelTensorShape const &input) {
  return solve_parallel_dim_mappings(construct_mappings(input), {input}, 0, 1);
}

ParallelTensorShape Pool2DParams::calculate_output_shape(ParallelTensorShape const &input) const {
  return solve_mappings(input).output_shapes.at(0);
}

typename Pool2DParams::AsConstTuple Pool2DParams::as_tuple() const {
  return {this->kernel_h, this->kernel_w, this->stride_h, this->stride_w, this->padding_h, this->padding_w, this->pool_type, this->activation};
}

bool operator==(Pool2DParams const &lhs, Pool2DParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(Pool2DParams const &lhs, Pool2DParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}
}

namespace std {
size_t hash<FlexFlow::Pool2DParams>::operator()(
    FlexFlow::Pool2DParams const &params) const {
  return get_std_hash(params.as_tuple()); 
}
}
