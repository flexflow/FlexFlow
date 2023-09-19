#ifndef _FLEXFLOW_SUBSTITUTIONS_OPERATOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_OPERATOR_PATTERN_H

#include "attribute_expr.h"
#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "pcg/operator.h"
#include <unordered_set>
#include <vector>

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
  AGGR,
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
  EMBED_DIM,
  KDIM,
  VDIM,
  DROPOUT,
  BIAS,
  ADD_BIAS_KV,
  ADD_ZERO_ATTN,
  A_SEQ_LENGTH_DIM,
  B_SEQ_LENGTH_DIM,
  RELU,
  TARGET_DIMS,
  RATE,
  SEED,
  SHOULD_BROADCAST_LHS,
  SHOULD_BROADCAST_RHS,
  DIM,
  ELEMENTWISE_AFFINE,
  REGULARIZER,
  SHAPE,
  SPLITS,
  K,
  SORTED,
  COMBINE_DIM,
  COMBINE_DEGREE,
};

using OperatorAttributeValue = variant<int,
                                       float,
                                       bool,
                                       stack_vector<int, MAX_TENSOR_DIM>,
                                       stack_vector<int, MAX_NUM_OUTPUTS>,
                                       OperatorType,
                                       Activation,
                                       ff_dim_t,
                                       unsigned long long,
                                       AggregateOp,
                                       stack_vector<ff_dim_t, MAX_TENSOR_DIM>,
                                       RegularizerAttrs,
                                       PoolOp,
                                       TensorShape,
                                       DataType>;

FF_VISITABLE_STRUCT(ListIndexAccess<FlexFlow::OperatorAttributeKey>,
                    attribute_key,
                    index);
FF_VISITABLE_STRUCT(ListSize<FlexFlow::OperatorAttributeKey>, attribute_key);

using OperatorAttributeConstraint =
    AttributeConstraint<OperatorAttributeKey, OperatorAttributeValue>;

using OperatorPattern =
    AttributePattern<OperatorAttributeKey, OperatorAttributeValue>;

optional<OperatorAttributeValue>
    evaluate_attribute_expr(Operator const &attrs,
                            AttributeExpr<OperatorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
