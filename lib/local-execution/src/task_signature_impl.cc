#include "local-execution/task_signature_impl.h"
#include "local-execution/ops/attention.h"
#include "ops/batch_matmul.h"
#include "ops/batch_norm.h"
#include "ops/cast.h"
#include "ops/combine.h"
#include "ops/concat.h"
#include "ops/conv_2d.h"
#include "ops/dropout.h"
#include "ops/element_binary.h"
#include "ops/element_unary.h"
#include "ops/embedding.h"
#include "ops/flat.h"
#include "ops/gather.h"
#include "ops/input.h"
#include "ops/layer_norm.h"
#include "ops/linear.h"
#include "ops/noop.h"
#include "ops/pool_2d.h"
#include "ops/reduce.h"
#include "ops/reduction.h"
#include "ops/repartition.h"
#include "ops/replicate.h"
#include "ops/reshape.h"
#include "ops/reverse.h"
#include "ops/softmax.h"
#include "ops/split.h"
#include "ops/topk.h"
#include "ops/transpose.h"
#include "ops/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

TaskSignatureAndImpl get_task_sig_impl(task_id_t const &task_id) {
  switch (task_id) {
    case task_id_t::ELEMENTBINARY_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_element_binary_init_task_impl(),
                                  get_element_binary_init_signature()};
    case task_id_t::ELEMENTBINARY_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_element_binary_fwd_task_impl(),
                                  get_element_binary_fwd_signature()};
    case task_id_t::ELEMENTBINARY_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_element_binary_bwd_task_impl(),
                                  get_element_binary_bwd_signature()};
    case task_id_t::ELEMENTUNARY_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_element_unary_init_task_impl(),
                                  get_element_unary_init_signature()};
    case task_id_t::ELEMENTUNARY_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_element_unary_fwd_task_impl(),
                                  get_element_unary_fwd_signature()};
    case task_id_t::ELEMENTUNARY_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_element_binary_bwd_task_impl(),
                                  get_element_binary_bwd_signature()};
    case task_id_t::CONV2D_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_conv_2d_init_task_impl(),
                                  get_conv_2d_init_signature()};
    case task_id_t::CONV2D_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_conv_2d_fwd_task_impl(),
                                  get_conv_2d_fwd_signature()};
    case task_id_t::CONV2D_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_conv_2d_bwd_task_impl(),
                                  get_conv_2d_bwd_signature()};
    case task_id_t::DROPOUT_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_dropout_init_task_impl(),
                                  get_dropout_init_signature()};
    case task_id_t::DROPOUT_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_dropout_fwd_task_impl(),
                                  get_dropout_fwd_signature()};
    case task_id_t::DROPOUT_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_dropout_bwd_task_impl(),
                                  get_dropout_bwd_signature()};
    // case task_id_t::EMBED_FWD_TASK_ID:
    //   return TaskSignatureAndImpl{get_embedding_fwd_task_impl(),
    //   get_embedding_fwd_signature()};
    // case task_id_t::EMBED_BWD_TASK_ID:
    //   return TaskSignatureAndImpl{get_embedding_bwd_task_impl(),
    //   get_embedding_bwd_signature()};
    case task_id_t::GATHER_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_gather_init_task_impl(),
                                  get_gather_init_signature()};
    case task_id_t::GATHER_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_gather_fwd_task_impl(),
                                  get_gather_fwd_signature()};
    case task_id_t::GATHER_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_gather_bwd_task_impl(),
                                  get_gather_bwd_signature()};
    case task_id_t::CAST_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_cast_fwd_task_impl(),
                                  get_cast_fwd_signature()};
    case task_id_t::CAST_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_cast_bwd_task_impl(),
                                  get_cast_bwd_signature()};
    case task_id_t::POOL2D_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_pool_2d_init_task_impl(),
                                  get_pool_2d_init_signature()};
    case task_id_t::POOL2D_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_pool_2d_fwd_task_impl(),
                                  get_pool_2d_fwd_signature()};
    case task_id_t::POOL2D_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_pool_2d_bwd_task_impl(),
                                  get_pool_2d_bwd_signature()};
    case task_id_t::BATCHNORM_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_batch_norm_init_task_impl(),
                                  get_batch_norm_init_signature()};
    case task_id_t::BATCHNORM_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_batch_norm_fwd_task_impl(),
                                  get_batch_norm_fwd_signature()};
    case task_id_t::BATCHNORM_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_batch_norm_bwd_task_impl(),
                                  get_batch_norm_bwd_signature()};
    case task_id_t::BATCHMATMUL_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_batch_matmul_fwd_task_impl(),
                                  get_batch_matmul_fwd_signature()};
    case task_id_t::BATCHMATMUL_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_batch_matmul_bwd_task_impl(),
                                  get_batch_matmul_bwd_signature()};
    case task_id_t::LAYERNORM_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_layer_norm_init_task_impl(),
                                  get_layer_norm_init_signature()};
    case task_id_t::LAYERNORM_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_layer_norm_fwd_task_impl(),
                                  get_layer_norm_init_signature()};
    case task_id_t::LAYERNORM_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_layer_norm_bwd_task_impl(),
                                  get_layer_norm_bwd_signature()};
    case task_id_t::LINEAR_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_linear_init_task_impl(),
                                  get_linear_init_signature()};
    case task_id_t::LINEAR_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_linear_fwd_task_impl(),
                                  get_linear_fwd_signature()};
    case task_id_t::LINEAR_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_linear_bwd_task_impl(),
                                  get_linear_bwd_signature()};
    case task_id_t::FLAT_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_flat_fwd_task_impl(),
                                  get_flat_fwd_signature()};
    case task_id_t::FLAT_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_flat_bwd_task_impl(),
                                  get_flat_bwd_signature()};
    case task_id_t::SOFTMAX_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_softmax_init_task_impl(),
                                  get_softmax_init_signature()};
    case task_id_t::SOFTMAX_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_softmax_fwd_task_impl(),
                                  get_softmax_fwd_signature()};
    case task_id_t::SOFTMAX_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_softmax_bwd_task_impl(),
                                  get_softmax_bwd_signature()};
    case task_id_t::CONCAT_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_concat_fwd_task_impl(),
                                  get_concat_fwd_signature()};
    case task_id_t::CONCAT_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_concat_bwd_task_impl(),
                                  get_concat_bwd_signature()};
    case task_id_t::SPLIT_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_split_fwd_task_impl(),
                                  get_split_fwd_signature()};
    case task_id_t::SPLIT_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_split_bwd_task_impl(),
                                  get_split_bwd_signature()};
    case task_id_t::REDUCE_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_reduce_init_task_impl(),
                                  get_reduce_init_signature()};
    case task_id_t::REDUCE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_reduce_fwd_task_impl(),
                                  get_reduce_fwd_signature()};
    case task_id_t::REDUCE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_reduce_bwd_task_impl(),
                                  get_reduce_bwd_signature()};
    case task_id_t::RESHAPE_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_reshape_init_task_impl(),
                                  get_reshape_init_signature()};
    case task_id_t::RESHAPE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_reshape_fwd_task_impl(),
                                  get_reshape_fwd_signature()};
    case task_id_t::RESHAPE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_reshape_bwd_task_impl(),
                                  get_reshape_bwd_signature()};
    case task_id_t::REVERSE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_reverse_fwd_task_impl(),
                                  get_reverse_fwd_signature()};
    case task_id_t::REVERSE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_reverse_bwd_task_impl(),
                                  get_reverse_bwd_signature()};
    case task_id_t::TOPK_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_topk_init_task_impl(),
                                  get_topk_init_signature()};
    case task_id_t::TOPK_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_topk_fwd_task_impl(),
                                  get_topk_fwd_signature()};
    case task_id_t::TOPK_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_topk_bwd_task_impl(),
                                  get_topk_bwd_signature()};
    case task_id_t::TRANSPOSE_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_transpose_init_task_impl(),
                                  get_transpose_init_signature()};
    case task_id_t::TRANSPOSE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_transpose_fwd_task_impl(),
                                  get_transpose_fwd_signature()};
    case task_id_t::TRANSPOSE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_transpose_bwd_task_impl(),
                                  get_transpose_bwd_signature()};
    case task_id_t::ATTENTION_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_attention_init_task_impl(),
                                  get_attention_init_signature()};
    case task_id_t::ATTENTION_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_attention_fwd_task_impl(),
                                  get_attention_fwd_signature()};
    case task_id_t::ATTENTION_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_attention_bwd_task_impl(),
                                  get_attention_bwd_signature()};
    case task_id_t::COMBINE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_combine_fwd_task_impl(),
                                  get_combine_fwd_signature()};
    case task_id_t::COMBINE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_combine_bwd_task_impl(),
                                  get_combine_bwd_signature()};
    case task_id_t::REDUCTION_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_reduction_fwd_task_impl(),
                                  get_reduction_fwd_signature()};
    case task_id_t::REDUCTION_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_reduction_bwd_task_impl(),
                                  get_reduction_bwd_signature()};
    case task_id_t::REPARTITION_INIT_TASK_ID:
      return TaskSignatureAndImpl{get_repartition_init_task_impl(),
                                  get_repartition_init_signature()};
    case task_id_t::REPARTITION_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_repartition_fwd_task_impl(),
                                  get_repartition_fwd_signature()};
    case task_id_t::REPARTITION_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_repartition_bwd_task_impl(),
                                  get_repartition_bwd_signature()};
    case task_id_t::REPLICATE_FWD_TASK_ID:
      return TaskSignatureAndImpl{get_replicate_fwd_task_impl(),
                                  get_replicate_fwd_signature()};
    case task_id_t::REPLICATE_BWD_TASK_ID:
      return TaskSignatureAndImpl{get_replicate_bwd_task_impl(),
                                  get_replicate_bwd_signature()};
    default:
      throw mk_runtime_error(
          fmt::format("Invalid task ID")); // inserting task_id yields
                                           // "type_is_unformattable" error
  }
}

std::vector<task_id_t> get_task_ids(ComputationGraphOpAttrs const &op) {
  return op.visit<std::vector<task_id_t>>(overload{
      [](BatchMatmulAttrs const &attrs) { return get_task_ids(attrs); },
      [](BatchNormAttrs const &attrs) { return get_task_ids(attrs); },
      [](CastAttrs const &attrs) { return get_task_ids(attrs); },
      [](ConcatAttrs const &attrs) { return get_task_ids(attrs); },
      [](Conv2DAttrs const &attrs) { return get_task_ids(attrs); },
      [](DropoutAttrs const &attrs) { return get_task_ids(attrs); },
      [](ElementBinaryAttrs const &attrs) { return get_task_ids(attrs); },
      [](ElementUnaryAttrs const &attrs) { return get_task_ids(attrs); },
      // [](EmbeddingAttrs const & attrs) {
      //   return get_task_ids(attrs);
      // },
      [](FlatAttrs const &attrs) { return get_task_ids(attrs); },
      [](GatherAttrs const &attrs) { return get_task_ids(attrs); },
      [](InputAttrs const &attrs) { return get_task_ids(attrs); },
      [](LayerNormAttrs const &attrs) { return get_task_ids(attrs); },
      [](LinearAttrs const &attrs) { return get_task_ids(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return get_task_ids(attrs); },
      [](NoopAttrs const &attrs) { return get_task_ids(attrs); },
      [](Pool2DAttrs const &attrs) { return get_task_ids(attrs); },
      [](ReduceAttrs const &attrs) { return get_task_ids(attrs); },
      [](ReverseAttrs const &attrs) { return get_task_ids(attrs); },
      [](ReshapeAttrs const &attrs) { return get_task_ids(attrs); },
      [](SplitAttrs const &attrs) { return get_task_ids(attrs); },
      [](SoftmaxAttrs const &attrs) { return get_task_ids(attrs); },
      [](TopKAttrs const &attrs) { return get_task_ids(attrs); },
      [](TransposeAttrs const &attrs) { return get_task_ids(attrs); },
      [](WeightAttrs const &attrs) { return get_task_ids(attrs); },
      [](auto const &attrs) -> std::vector<task_id_t> {
        throw mk_runtime_error(fmt::format("Unhandled attr type: {}", attrs));
      },
  });
}

OpTaskInvocation init(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchNormAttrs const &attrs) { return init(attrs); },
      [](Conv2DAttrs const &attrs) { return init(attrs); },
      [](DropoutAttrs const &attrs) { return init(attrs); },
      [](ElementBinaryAttrs const &attrs) { return init(attrs); },
      [](ElementUnaryAttrs const &attrs) { return init(attrs); },
      [](GatherAttrs const &attrs) { return init(attrs); },
      [](LayerNormAttrs const &attrs) { return init(attrs); },
      [](LinearAttrs const &attrs) { return init(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return init(attrs); },
      [](Pool2DAttrs const &attrs) { return init(attrs); },
      [](ReduceAttrs const &attrs) { return init(attrs); },
      [](ReshapeAttrs const &attrs) { return init(attrs); },
      [](SoftmaxAttrs const &attrs) { return init(attrs); },
      [](TopKAttrs const &attrs) { return init(attrs); },
      [](TransposeAttrs const &attrs) { return init(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        throw mk_runtime_error(fmt::format("Unhandled attr type {}", attrs));
      },
  });
}

OpTaskInvocation forward(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchMatmulAttrs const &attrs) { return forward(attrs); },
      [](BatchNormAttrs const &attrs) { return forward(attrs); },
      [](CastAttrs const &attrs) { return forward(attrs); },
      [](ConcatAttrs const &attrs) { return forward(attrs); },
      [](Conv2DAttrs const &attrs) { return forward(attrs); },
      [](DropoutAttrs const &attrs) { return forward(attrs); },
      [](ElementBinaryAttrs const &attrs) { return forward(attrs); },
      [](ElementUnaryAttrs const &attrs) { return forward(attrs); },
      // [](EmbeddingAttrs const & attrs) {
      //   return forward(attrs);
      // },
      [](FlatAttrs const &attrs) { return forward(attrs); },
      [](GatherAttrs const &attrs) { return forward(attrs); },
      [](LayerNormAttrs const &attrs) { return forward(attrs); },
      [](LinearAttrs const &attrs) { return forward(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return forward(attrs); },
      [](Pool2DAttrs const &attrs) { return forward(attrs); },
      [](ReduceAttrs const &attrs) { return forward(attrs); },
      [](ReverseAttrs const &attrs) { return forward(attrs); },
      [](ReshapeAttrs const &attrs) { return forward(attrs); },
      [](SplitAttrs const &attrs) { return forward(attrs); },
      [](SoftmaxAttrs const &attrs) { return forward(attrs); },
      [](TopKAttrs const &attrs) { return forward(attrs); },
      [](TransposeAttrs const &attrs) { return forward(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        throw mk_runtime_error(fmt::format("Unhandled attr type {}", attrs));
      },
  });
}

OpTaskInvocation backward(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchMatmulAttrs const &attrs) { return backward(attrs); },
      [](BatchNormAttrs const &attrs) { return backward(attrs); },
      [](CastAttrs const &attrs) { return backward(attrs); },
      [](ConcatAttrs const &attrs) { return backward(attrs); },
      [](Conv2DAttrs const &attrs) { return backward(attrs); },
      [](DropoutAttrs const &attrs) { return backward(attrs); },
      [](ElementBinaryAttrs const &attrs) { return backward(attrs); },
      [](ElementUnaryAttrs const &attrs) { return backward(attrs); },
      // [](EmbeddingAttrs const & attrs) {
      //   return backward(attrs);
      // },
      [](FlatAttrs const &attrs) { return backward(attrs); },
      [](GatherAttrs const &attrs) { return backward(attrs); },
      [](LayerNormAttrs const &attrs) { return backward(attrs); },
      [](LinearAttrs const &attrs) { return backward(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return backward(attrs); },
      [](Pool2DAttrs const &attrs) { return backward(attrs); },
      [](ReduceAttrs const &attrs) { return backward(attrs); },
      [](ReverseAttrs const &attrs) { return backward(attrs); },
      [](ReshapeAttrs const &attrs) { return backward(attrs); },
      [](SplitAttrs const &attrs) { return backward(attrs); },
      [](SoftmaxAttrs const &attrs) { return backward(attrs); },
      [](TopKAttrs const &attrs) { return backward(attrs); },
      [](TransposeAttrs const &attrs) { return backward(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        throw mk_runtime_error(fmt::format("Unhandled attr type {}", attrs));
      },
  });
}

} // namespace FlexFlow
