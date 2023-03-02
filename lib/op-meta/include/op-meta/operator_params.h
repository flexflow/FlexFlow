#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "op-meta/ops/aggregate_params.h"
#include "op-meta/ops/aggregate_spec_params.h"
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
#include "op-meta/ops/gather_params.h"
#include "op-meta/ops/groupby_params.h"
#include "op-meta/ops/layer_norm_params.h"
#include "op-meta/ops/linear_params.h"
#include "op-meta/ops/pool_2d_params.h"
#include "op-meta/ops/reshape_params.h"
#include "op-meta/ops/softmax_params.h"
#include "op-meta/ops/split_params.h"
#include "op-meta/ops/transpose_params.h"
#include "op-meta/ops/combine_params.h"
#include "op-meta/ops/fused_parallel_op_params.h"
#include "op-meta/ops/repartition_params.h"
#include "op-meta/ops/reduce_params.h"
#include "op-meta/ops/reduction_params.h"
#include "op-meta/ops/replicate_params.h"
#include "op-meta/ops/topk_params.h"
#include "mpark/variant.hpp"

namespace FlexFlow {
namespace opmeta {

// TODO: Inherit these operators from OpParamsInterface. Comment out temporarily to avoid compile error.
using OperatorParameters = mpark::variant<AggregateParams,
                                       AggregateSpecParams,
                                       BatchMatmulParams,
                                       CastParams,
                                       CombineParams,
                                       ConcatParams,
                                       Conv2DParams,
                                       DropoutParams,
                                       ElementBinaryParams,
                                       ElementUnaryParams,
                                       EmbeddingParams,
                                       FlatParams,
                                       GatherParams,
                                       Group_byParams,
                                       LayerNormParams,
                                       LinearParams,
                                       MultiHeadAttentionParams,
                                       Pool2DParams,
                                       ReduceParams,
                                       ReductionParams,
                                       RepartitionParams,
                                       ReplicateParams,
                                       ReshapeParams,
                                       SplitParams,
                                       SoftmaxParams,
                                       TopKParams,
                                       TransposeParams,
                                       FusedParallelOpParams>;

OperatorType get_op_type(OperatorParameters const &);
OperatorType get_op_type(OpParamsInterface const &);
RecordFormatter as_dot(OperatorParameters const &);

std::vector<ParallelTensorShape> get_output_shapes(OperatorParameters const &op_params,
                                                   std::vector<ParallelTensorShape> const &input_tensor_shapes);

bool is_parallel_op(OperatorParameters const &);

}; // namespace FlexFlow
};

#endif // _OPERATOR_PARAMS_H
