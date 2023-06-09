#ifndef _FLEXFLOW_OP_META_SRC_PARELLEL_DIM_MAPPING_RECORD_H
#define _FLEXFLOW_OP_META_SRC_PARELLEL_DIM_MAPPING_RECORD_H

#include "tl/optional.hpp"
#include "utils/visitable.h"
#include <vector>

namespace FlexFlow {

enum class MappingRecordType { INPUT_OUTPUT, INPUT_WEIGHT };

enum class MappingOperation { PARTITION, REPLICATE };

class ParallelDimMappingRecord {
private:
  ParallelDimMappingRecord(MappingRecordType);

public:
  ParallelDimMappingRecord() = delete;

  static ParallelDimMappingRecord input_output_record(
      int input_idx,
      int input_dim,
      int output_idx,
      int output_dim,
      tl::optional<MappingOperation> operation = tl::nullopt);
  static ParallelDimMappingRecord input_weight_record(
      int input_idx,
      int input_dim,
      int weight_idx,
      int weight_dim,
      tl::optional<MappingOperation> operation = tl::nullopt);
  MappingRecordType get_type() const;

public:
  MappingRecordType type;
  tl::optional<MappingOperation> operation;

  int output_dim, input_dim, weight_dim;
  int output_idx, input_idx, weight_idx;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ParallelDimMappingRecord,
                 type,
                 operation,
                 output_dim,
                 input_dim,
                 weight_dim,
                 output_idx,
                 input_idx,
                 weight_idx);

#endif
