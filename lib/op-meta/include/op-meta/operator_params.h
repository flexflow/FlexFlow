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
#include "op-meta/ops/repartition_params.h"
#include "op-meta/ops/reduction_params.h"
#include "op-meta/ops/replicate_params.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace FlexFlow {
namespace opmeta {

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

OperatorType get_op_type(OperatorParameters const &);
OperatorType get_op_type(OpParamsInterface const &);

bool is_parallel_op(OperatorParameters const &);

}; // namespace FlexFlow
};

#endif // _OPERATOR_PARAMS_H
