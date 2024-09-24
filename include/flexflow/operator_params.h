#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "flexflow/ops/add_bias_residual_layer_norm_params.h"
#include "flexflow/ops/aggregate_params.h"
#include "flexflow/ops/aggregate_spec_params.h"
#include "flexflow/ops/arg_topk_params.h"
#include "flexflow/ops/argmax_params.h"
#include "flexflow/ops/attention_params.h"
#include "flexflow/ops/batch_matmul_params.h"
#include "flexflow/ops/beam_topk_params.h"
#include "flexflow/ops/cast_params.h"
#include "flexflow/ops/concat_params.h"
#include "flexflow/ops/conv_2d_params.h"
#include "flexflow/ops/dropout_params.h"
#include "flexflow/ops/element_binary_params.h"
#include "flexflow/ops/element_unary_params.h"
#include "flexflow/ops/embedding_params.h"
#include "flexflow/ops/experts_params.h"
#include "flexflow/ops/flat_params.h"
#include "flexflow/ops/gather_params.h"
#include "flexflow/ops/groupby_params.h"
#include "flexflow/ops/inc_multihead_self_attention_params.h"
#include "flexflow/ops/layer_norm_params.h"
#include "flexflow/ops/linear_params.h"
#include "flexflow/ops/lora_linear_params.h"
#include "flexflow/ops/pool_2d_params.h"
#include "flexflow/ops/reduce_params.h"
#include "flexflow/ops/reshape_params.h"
#include "flexflow/ops/residual_layer_norm_params.h"
#include "flexflow/ops/residual_rms_norm_params.h"
#include "flexflow/ops/rms_norm_params.h"
#include "flexflow/ops/sampling_params.h"
#include "flexflow/ops/sigmoid_silu_multi_params.h"
#include "flexflow/ops/softmax_params.h"
#include "flexflow/ops/spec_inc_multihead_self_attention_params.h"
#include "flexflow/ops/split_params.h"
#include "flexflow/ops/topk_params.h"
#include "flexflow/ops/transpose_params.h"
#include "flexflow/ops/tree_inc_multihead_self_attention_params.h"
#include "flexflow/parallel_ops/allreduce_params.h"
#include "flexflow/parallel_ops/combine_params.h"
#include "flexflow/parallel_ops/fused_parallel_op_params.h"
#include "flexflow/parallel_ops/parallel_identity_params.h"
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
                                       ResidualLayerNormParams,
                                       AddBiasResidualLayerNormParams,
                                       SigmoidSiluMultiParams,
                                       LinearParams,
                                       LoraLinearParams,
                                       MultiHeadAttentionParams,
                                       IncMultiHeadSelfAttentionParams,
                                       BeamTopKParams,
                                       SpecIncMultiHeadSelfAttentionParams,
                                       TreeIncMultiHeadSelfAttentionParams,
                                       RMSNormParams,
                                       ResidualRMSNormParams,
                                       Pool2DParams,
                                       ReduceParams,
                                       ReshapeParams,
                                       SplitParams,
                                       TopKParams,
                                       ArgTopKParams,
                                       SamplingParams,
                                       ArgMaxParams,
                                       SoftmaxParams,
                                       TransposeParams,
                                       RepartitionParams,
                                       ReplicateParams,
                                       ReductionParams,
                                       CombineParams,
                                       AllReduceParams,
                                       ParallelIdentityParams,
                                       FusedParallelOpParams>;

tl::optional<OperatorParameters> get_op_parameters(Op const *op);

}; // namespace FlexFlow

#endif // _OPERATOR_PARAMS_H
