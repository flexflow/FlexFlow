#ifndef _FLEXFLOW_SUBSTITUTIONS_OPERATOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_OPERATOR_PATTERN_H

#include "constraint.h"

namespace FlexFlow {

enum class OperatorAttributeKey {
  OP_TYPE, // AnyOp
  USE_BIAS,
  GROUPS,
  POOL_TYPE,
  KERNEL_H,
  KERNEL_W,
  DATA_TYPE,
  SCALAR,
  STRIDE_H,
  STRIDE_W,
  PADDING_H,
  PADDING_W,
  AGGR_MODE,
  NUM_ENTRIES,
  OUT_CHANNELS,
  ACTIVATION,
  NUMDIM,
  AXIS,
  PERMUTATION,
  OUTSHUFFLE,
  MERGE_GCONV_COUNT,
  AXES,
  KEEP_DIMS,
  EPSILON,
  PARALLEL_OP_DIM,
  PARALLEL_OP_DEGREE,
  SOFTMAX_DIM,
  NUM_HEADS,
  PARALLEL_DIM,
  PARALLEL_DEGREE,
  PAD,
};

using OperatorAttributeValue =
    variant<int, float, bool, std::vector<int>, OperatorType, Activation>;

using OperatorAttributeConstraint =
    AttributeConstraint<OperatorAttributeKey, OperatorAttributeValue>;

struct OperatorPattern {
  std::unordered_set<OperatorAttributeConstraint> attribute_constraints;
};

} // namespace FlexFlow

#endif
