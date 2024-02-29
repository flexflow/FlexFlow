#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "flexflow/ops/aggregate_params.h"
#include "flexflow/ops/aggregate_spec_params.h"
#include "flexflow/ops/attention_params.h"
#include "flexflow/ops/batch_matmul_params.h"
#include "flexflow/ops/cast_params.h"
#include "flexflow/ops/concat_params.h"
#include "flexflow/parallel_ops/allreduce_params.h"
#include "flexflow/ops/conv_2d_params.h"
#include "flexflow/ops/dropout_params.h"
#include "flexflow/ops/element_binary_params.h"
#include "flexflow/ops/element_unary_params.h"
#include "flexflow/ops/embedding_params.h"
#include "flexflow/ops/flat_params.h"
#include "flexflow/ops/gather_params.h"
#include "flexflow/ops/groupby_params.h"
#include "flexflow/ops/layer_norm_params.h"
#include "flexflow/ops/linear_params.h"
#include "flexflow/ops/pool_2d_params.h"
#include "flexflow/ops/reduce_params.h"
#include "flexflow/ops/reshape_params.h"
#include "flexflow/ops/softmax_params.h"
#include "flexflow/ops/split_params.h"
#include "flexflow/ops/topk_params.h"
#include "flexflow/ops/transpose_params.h"
#include "flexflow/parallel_ops/combine_params.h"
#include "flexflow/parallel_ops/fused_parallel_op_params.h"
#include "flexflow/parallel_ops/partition_params.h"
#include "flexflow/parallel_ops/reduction_params.h"
#include "flexflow/parallel_ops/replicate_params.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace FlexFlow {

using OperatorParameters = mp::variant<AggregateParams,
                                       AggregateSpecParams,
                                       BatchMatmulParams,
                                       Conv2DParams,
                                       ConcatParams,
                                       CastParams,
                                       ElementBinaryParams,
                                       ElementUnaryParams,
                                       DropoutParams,
                                       EmbeddingParams,
                                       FlatParams,
                                       GatherParams,
                                       Group_byParams,
                                       LayerNormParams,
                                       LinearParams,
                                       MultiHeadAttentionParams,
                                       Pool2DParams,
                                       ReduceParams,
                                       ReshapeParams,
                                       SplitParams,
                                       TopKParams,
                                       SoftmaxParams,
                                       TransposeParams,
                                       RepartitionParams,
                                       ReplicateParams,
                                       ReductionParams,
                                       CombineParams,
                                       AllReduceParams,
                                       FusedParallelOpParams>;

tl::optional<OperatorParameters> get_op_parameters(Op const *op);

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
