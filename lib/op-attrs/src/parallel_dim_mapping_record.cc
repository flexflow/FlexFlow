#include "parallel_dim_mapping_record.h"
#include <cassert>

namespace FlexFlow {

ParallelDimMappingRecord::ParallelDimMappingRecord(MappingRecordType type)
    : type(type), output_dim(-1), input_dim(-1), weight_dim(-1), output_idx(-1),
      input_idx(-1), weight_idx(-1) {}

/*static*/
ParallelDimMappingRecord ParallelDimMappingRecord::input_output_record(
    int input_idx,
    int input_dim,
    int output_idx,
    int output_dim,
    tl::optional<MappingOperation> operation) {
  ParallelDimMappingRecord r(MappingRecordType::INPUT_OUTPUT);
  r.operation = operation;

  assert(output_idx >= 0);
  assert(output_dim >= 0);
  assert(input_idx >= 0);
  assert(input_dim >= 0);

  r.output_idx = output_idx;
  r.output_dim = output_dim;
  r.input_idx = input_idx;
  r.input_dim = input_dim;

  return r;
}

/*static*/
ParallelDimMappingRecord ParallelDimMappingRecord::input_weight_record(
    int input_idx,
    int input_dim,
    int weight_idx,
    int weight_dim,
    tl::optional<MappingOperation> operation) {
  ParallelDimMappingRecord r(MappingRecordType::INPUT_WEIGHT);
  r.operation = operation;

  assert(input_idx >= 0);
  assert(input_dim >= 0);
  assert(weight_idx >= 0);
  assert(weight_dim >= 0);

  r.input_idx = input_idx;
  r.input_dim = input_dim;
  r.weight_idx = weight_idx;
  r.weight_dim = weight_dim;

  return r;
}

MappingRecordType ParallelDimMappingRecord::get_type() const {
  return this->type;
}

} // namespace FlexFlow
