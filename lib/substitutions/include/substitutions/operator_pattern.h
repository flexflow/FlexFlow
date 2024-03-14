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

/**
 * @enum OperatorAttributeKey
 * @brief OperatorAttributeKey represents the keys of the attributes of an Operator.
 * Specifically, each operator have a set of attributes, and each attribute will have 
 * a key as its name and a concrete value representation.
 * The OP_TYPE is a OperatorAttributeKey is a special attribute key that represents the 
 * type of the Operator and will exist in every Operator. Given the OP_TYPE, the other 
 * attributes will be determined accordingly.
 * 
 * For example, a batch matrix multiplication Operator will have OP_TYPE BATCH_MATMUL and 
 * dimensions as A_SEQ_LENGTH_DIM and B_SEQ_LENGTH_DIM
 */
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
  NUM_INPUTS
};

/**
 * @brief OperatorAttributeValue is a representation of the concrete value of an attribute of an Operator.
 * The OperatorAttributeValue is evaluated from AttributeExpr. The datatype of the value corresponds to the 
 * datatype of the attributekey listed in OperatorAttributeKey.
 */
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
                                       optional<RegularizerAttrs>,
                                       PoolOp,
                                       TensorShape,
                                       DataType>;

FF_VISITABLE_STRUCT(ListIndexAccess<FlexFlow::OperatorAttributeKey>,
                    attribute_key,
                    index);
FF_VISITABLE_STRUCT(ListSize<FlexFlow::OperatorAttributeKey>, attribute_key);

/**
 * @brief OperatorAttributeConstraint is an instance of template struct AttributeConstraint.
 */
using OperatorAttributeConstraint =
    AttributeConstraint<OperatorAttributeKey, OperatorAttributeValue>;

/**
 * @brief OperatorPattern is an instance of template struct AttributePattern.
 */
using OperatorPattern =
    AttributePattern<OperatorAttributeKey, OperatorAttributeValue>;


/**
 * @brief Given a specific attribute of an Operator, evaluate the expression of the attribute 
 * using one of the three methods: direct value, list index access, or list size and return the
 * value of the attribute.
 */
optional<OperatorAttributeValue>
    evaluate_attribute_expr(Operator const &attrs,
                            AttributeExpr<OperatorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
