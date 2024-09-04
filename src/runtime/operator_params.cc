#include "flexflow/operator_params.h"
#include "flexflow/ops/add_bias_residual_layer_norm.h"
#include "flexflow/ops/aggregate.h"
#include "flexflow/ops/aggregate_spec.h"
#include "flexflow/ops/arg_topk.h"
#include "flexflow/ops/argmax.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/batch_matmul.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/beam_topk.h"
#include "flexflow/ops/cache.h"
#include "flexflow/ops/cast.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/gather.h"
#include "flexflow/ops/groupby.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/layer_norm.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/mean.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reduce.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/residual_layer_norm.h"
#include "flexflow/ops/residual_rms_norm.h"
#include "flexflow/ops/reverse.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/ops/sampling.h"
#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/ops/split.h"
#include "flexflow/ops/topk.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/parallel_ops/allreduce.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/parallel_ops/parallel_identity.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/parallel_ops/replicate.h"

namespace FlexFlow {

tl::optional<OperatorParameters> get_op_parameters(Op const *op) {
  switch (op->op_type) {
    case OP_LINEAR:
      return ((Linear *)op)->get_params();
    case OP_CONV2D:
      return ((Conv2D *)op)->get_params();
    case OP_EW_ADD:
    case OP_EW_SUB:
    case OP_EW_MUL:
    case OP_EW_DIV:
    case OP_EW_MAX:
    case OP_EW_MIN:
      return ((ElementBinary *)op)->get_params();
    case OP_EXP:
    case OP_SIN:
    case OP_COS:
    case OP_SCALAR_MULTIPLY:
    case OP_SCALAR_ADD:
    case OP_SCALAR_SUB:
    case OP_SCALAR_TRUE_DIV:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    case OP_IDENTITY:
    case OP_GELU:
    case OP_ELU:
      return ((ElementUnary *)op)->get_params();
    case OP_CONCAT:
      return ((Concat *)op)->get_params();
    case OP_POOL2D:
      return ((Pool2D *)op)->get_params();
    case OP_CAST:
      return ((Cast *)op)->get_params();
    case OP_DROPOUT:
      return ((Dropout *)op)->get_params();
    case OP_EMBEDDING:
      return ((Embedding *)op)->get_params();
    case OP_FLAT:
      return ((Flat *)op)->get_params();
    case OP_GATHER:
      return ((Gather *)op)->get_params();
    case OP_MULTIHEAD_ATTENTION:
      return ((MultiHeadAttention *)op)->get_params();
    case OP_INC_MULTIHEAD_SELF_ATTENTION:
      return ((IncMultiHeadSelfAttention *)op)->get_params();
    case OP_TREE_INC_MULTIHEAD_SELF_ATTENTION:
      return ((TreeIncMultiHeadSelfAttention *)op)->get_params();
    case OP_LAYERNORM:
      return ((LayerNorm *)op)->get_params();
    case OP_RESIDUAL_LAYERNORM:
      return ((ResidualLayerNorm *)op)->get_params();
    case OP_ADD_BIAS_RESIDUAL_LAYERNORM:
      return ((AddBiasResidualLayerNorm *)op)->get_params();
    case OP_SIGMOID_SILU_MULTI:
      return ((SigmoidSiluMulti *)op)->get_params();
    case OP_REDUCE_SUM:
      return ((Reduce *)op)->get_params();
    case OP_RESHAPE:
      return ((Reshape *)op)->get_params();
    case OP_SOFTMAX:
      return ((Softmax *)op)->get_params();
    case OP_REPARTITION:
      return ((Repartition *)op)->get_params();
    case OP_REPLICATE:
      return ((Replicate *)op)->get_params();
    case OP_REDUCTION:
      return ((Reduction *)op)->get_params();
    case OP_COMBINE:
      return ((Combine *)op)->get_params();
    case OP_ALLREDUCE:
      return ((AllReduce *)op)->get_params();
    case OP_PARALLEL_IDENTITY:
      return ((ParallelIdentity *)op)->get_params();
    case OP_FUSED_PARALLEL:
      return ((FusedParallelOp *)op)->get_params();
    case OP_TRANSPOSE:
      return ((Transpose *)op)->get_params();
    case OP_BATCHMATMUL:
      return ((BatchMatmul *)op)->get_params();
    case OP_SPLIT:
      return ((Split *)op)->get_params();
    case OP_TOPK:
      return ((TopK *)op)->get_params();
    case OP_GROUP_BY:
      return ((Group_by *)op)->get_params();
    case OP_AGGREGATE:
      return ((Aggregate *)op)->get_params();
    case OP_AGG_SPEC:
      return ((AggregateSpec *)op)->get_params();
    case OP_RMS_NORM:
      return ((RMSNorm *)op)->get_params();
    case OP_RESIDUAL_RMS_NORM:
      return ((ResidualRMSNorm *)op)->get_params();
    case OP_ARG_TOPK:
      return ((ArgTopK *)op)->get_params();
    case OP_BEAM_TOPK:
      return ((BeamTopK *)op)->get_params();
    case OP_SAMPLING:
      return ((Sampling *)op)->get_params();
    case OP_ARGMAX:
      return ((ArgMax *)op)->get_params();

      // TODO: implement the get_params() function for the operators below and
      // uncomment the lines below

      // case OP_NOOP:
      //   return ((NoOp *)op)->get_params();
      // case OP_MEAN:
      //   return ((Mean *)op)->get_params();
      // case OP_CACHE:
      //   return ((Cache *)op)->get_params();
      // case OP_REVERSE:
      //   return ((Reverse *)op)->get_params();
      // case OP_BATCHNORM:
      //   return ((BatchNorm *)op)->get_params();

    default:
      return tl::nullopt;
  }
}

}; // namespace FlexFlow
