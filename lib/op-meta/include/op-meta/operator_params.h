#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "op-meta/ops/attention_params.h"
#include "op-meta/ops/batch_matmul_params.h"
#include "op-meta/ops/cast_params.h"
#include "op-meta/ops/concat_params.h"
#include "op-meta/ops/conv_2d_params.h"
#include "op-meta/ops/dropout_params.h"
#include "op-meta/ops/element_binary_params.h"
#include "op-meta/ops/element_unary_params.h"
#include "op-meta/ops/embedding_params.h"
#include "op-meta/ops/flat_params.h"
#include "op-meta/ops/layer_norm_params.h"
#include "op-meta/ops/linear_params.h"
#include "op-meta/ops/pool_2d_params.h"
#include "op-meta/ops/reshape_params.h"
#include "op-meta/ops/softmax_params.h"
#include "op-meta/ops/split_params.h"
#include "op-meta/ops/transpose_params.h"
#include "op-meta/ops/combine_params.h"
#include "op-meta/ops/fused_parallel_op_params.h"
#include "op-meta/ops/partition_params.h"
#include "op-meta/ops/reduction_params.h"
#include "op-meta/ops/replicate_params.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace FlexFlow {

using OperatorParameters = mp::variant<BatchMatmulParams,
                                       Conv2DParams,
                                       ConcatParams,
                                       CastParams,
                                       ElementBinaryParams,
                                       ElementUnaryParams,
                                       DropoutParams,
                                       EmbeddingParams,
                                       FlatParams,
                                       LayerNormParams,
                                       LinearParams,
                                       MultiHeadAttentionParams,
                                       Pool2DParams,
                                       ReshapeParams,
                                       SplitParams,
                                       SoftmaxParams,
                                       TransposeParams,
                                       RepartitionParams,
                                       ReplicateParams,
                                       ReductionParams,
                                       CombineParams,
                                       FusedParallelOpParams>;

struct GetOpType {
  OperatorType operator()(BatchMatmulParams const &p) const;
  OperatorType operator()(Conv2DParams const &p) const;
  OperatorType operator()(ConcatParams const &p) const;
  OperatorType operator()(CastParams const &p) const;
  OperatorType operator()(ElementBinaryParams const &p) const;
  OperatorType operator()(ElementUnaryParams const &p) const;
  OperatorType operator()(DropoutParams const &p) const;
  OperatorType operator()(EmbeddingParams const &p) const;
  OperatorType operator()(FlatParams const &p) const;
  OperatorType operator()(LayerNormParams const &p) const;
  OperatorType operator()(LinearParams const &p) const;
  OperatorType operator()(MultiHeadAttentionParams const &p) const;
  OperatorType operator()(Pool2DParams const &p) const;
  OperatorType operator()(ReshapeParams const &p) const;
  OperatorType operator()(SplitParams const &p) const;
  OperatorType operator()(SoftmaxParams const &p) const;
  OperatorType operator()(TransposeParams const &p) const;
  OperatorType operator()(RepartitionParams const &p) const;
  OperatorType operator()(ReplicateParams const &p) const;
  OperatorType operator()(ReductionParams const &p) const;
  OperatorType operator()(CombineParams const &p) const;
  OperatorType operator()(FusedParallelOpParams const &p) const;
};


template <typename T>
OperatorType get_op_type(T const &t) {
  return GetOpType{}(t);
};

template <>
OperatorType get_op_type(OperatorParameters const &);

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
