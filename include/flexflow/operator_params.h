#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "flexflow/ops/params/attention_params.h"
#include "flexflow/ops/params/batch_matmul_params.h"
#include "flexflow/ops/params/cast_params.h"
#include "flexflow/ops/params/concat_params.h"
#include "flexflow/ops/params/conv_2d_params.h"
#include "flexflow/ops/params/dropout_params.h"
#include "flexflow/ops/params/element_binary_params.h"
#include "flexflow/ops/params/element_unary_params.h"
#include "flexflow/ops/params/embedding_params.h"
#include "flexflow/ops/params/flat_params.h"
#include "flexflow/ops/params/layer_norm_params.h"
#include "flexflow/ops/params/linear_params.h"
#include "flexflow/ops/params/noop_params.h"
#include "flexflow/ops/params/pool_2d_params.h"
#include "flexflow/ops/params/reshape_params.h"
#include "flexflow/ops/params/softmax_params.h"
#include "flexflow/ops/params/split_params.h"
#include "flexflow/ops/params/transpose_params.h"
#include "flexflow/parallel_ops/params/combine_params.h"
#include "flexflow/parallel_ops/params/fused_parallel_op_params.h"
#include "flexflow/parallel_ops/params/partition_params.h"
#include "flexflow/parallel_ops/params/reduction_params.h"
#include "flexflow/parallel_ops/params/replicate_params.h"
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
                                       NoOpParams,
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

tl::optional<OperatorParameters> get_op_parameters(Op const *op);

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
