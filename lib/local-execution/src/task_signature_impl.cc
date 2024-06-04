#include "local-execution/task_signature_impl.h"
#include "ops/attention.h"
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
#include "ops/layer_norm.h"
#include "ops/linear.h"
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

namespace FlexFlow {

TaskSignatureAndImpl get_task_sig_impl(task_id_t const &task_id) {
  switch (task_id) {
    case ELEMENTBINARY_INIT_TASK_ID:
      return {get_element_binary_init_task_impl(),
              get_element_binary_init_signature()};
    case ELEMENTBINARY_FWD_TASK_ID:
      return {get_element_binary_fwd_task_impl(),
              get_element_binary_fwd_signature()};
    case ELEMENTBINARY_BWD_TASK_ID:
      return {get_element_binary_bwd_task_impl(),
              get_element_binary_bwd_signature()};
    case ELEMENTUNARY_INIT_TASK_ID:
      return {get_element_unary_init_task_impl(),
              get_element_unary_init_signature()};
    case ELEMENTUNARY_FWD_TASK_ID:
      return {get_element_unary_fwd_task_impl(),
              get_element_unary_fwd_signature()};
    case ELEMENTUNARY_BWD_TASK_ID:
      return {get_element_binary_bwd_task_impl(),
              get_element_binary_bwd_signature()};
    case CONV2D_INIT_TASK_ID:
      return {get_conv_2d_init_task_impl(), get_conv_2d_init_signature()};
    case CONV2D_FWD_TASK_ID:
      return {get_conv_2d_fwd_task_impl(), get_conv_2d_fwd_signature()};
    case CONV2D_BWD_TASK_ID:
      return {get_conv_2d_bwd_task_impl(), get_conv_2d_bwd_signature()};
    case DROPOUT_INIT_TASK_ID:
      return {get_dropout_init_task_impl(), get_dropout_init_signature()};
    case DROPOUT_FWD_TASK_ID:
      return {get_dropout_fwd_task_impl(), get_dropout_fwd_signature()};
    case DROPOUT_BWD_TASK_ID:
      return {get_dropout_bwd_task_impl(), get_dropout_bwd_signature()};
    case EMBED_FWD_TASK_ID:
      return {get_embedding_fwd_task_impl(), get_embedding_fwd_signature()};
    case EMBED_BWD_TASK_ID:
      return {get_embedding_bwd_task_impl(), get_embedding_bwd_signature()};
    case GATHER_INIT_TASK_ID:
      return {get_gather_init_task_impl(), get_gather_init_signature()};
    case GATHER_FWD_TASK_ID:
      return {get_gather_fwd_task_impl(), get_embedding_fwd_signature()};
    case GATHER_BWD_TASK_ID:
      return {get_gather_bwd_task_impl(), get_embedding_bwd_signature()};
    case CAST_FWD_TASK_ID:
      return {get_cast_fwd_task_impl(), get_cast_fwd_signature()};
    case CAST_BWD_TASK_ID:
      return {get_cast_bwd_task_impl(), get_cast_bwd_signature()};
    case POOL2D_INIT_TASK_ID:
      return {get_pool_2d_init_task_impl(), get_pool_2d_init_signature()};
    case POOL2D_FWD_TASK_ID:
      return {get_pool_2d_fwd_task_impl(), get_pool_2d_fwd_signature()};
    case POOL2D_BWD_TASK_ID:
      return {get_pool_2d_bwd_task_impl(), get_pool_2d_bwd_signature()};
    case BATCHNORM_INIT_TASK_ID:
      return {get_batch_norm_init_task_impl(), get_batch_norm_init_signature()};
    case BATCHNORM_FWD_TASK_ID:
      return {get_batch_norm_fwd_task_impl(), get_batch_norm_fwd_signature()};
    case BATCHNORM_BWD_TASK_ID:
      return {get_batch_norm_bwd_task_impl(), get_batch_norm_bwd_signature()};
    case BATCHMATMUL_FWD_TASK_ID:
      return {get_batch_matmul_fwd_task_impl(),
              get_batch_matmul_fwd_signature()};
    case BATCHMATMUL_BWD_TASK_ID:
      return {get_batch_matmul_bwd_task_impl(),
              get_batch_matmul_bwd_signature()};
    case LAYERNORM_INIT_TASK_ID:
      return {get_layer_norm_init_task_impl(), get_layer_norm_init_signature()};
    case LAYERNORM_FWD_TASK_ID:
      return {get_layer_norm_fwd_task_impl(), get_layer_norm_init_signature()};
    case LAYERNORM_BWD_TASK_ID:
      return {get_layer_norm_bwd_task_impl(), get_layer_norm_bwd_signature()};
    case LINEAR_INIT_TASK_ID:
      return {get_linear_init_task_impl(), get_linear_init_signature()};
    case LINEAR_FWD_TASK_ID:
      return {get_linear_fwd_task_impl(), get_linear_fwd_signature()};
    case LINEAR_BWD_TASK_ID:
      return {get_linear_bwd_task_impl(), get_linear_bwd_signature()};
    case FLAT_FWD_TASK_ID:
      return {get_flat_fwd_task_impl(), get_flat_fwd_signature()};
    case FLAT_BWD_TASK_ID:
      return {get_flat_bwd_task_impl(), get_flat_bwd_signature()};
    case SOFTMAX_INIT_TASK_ID:
      return {get_softmax_init_task_impl(), get_softmax_init_signature()};
    case SOFTMAX_FWD_TASK_ID:
      return {get_softmax_fwd_task_impl(), get_softmax_fwd_signature()};
    case SOFTMAX_BWD_TASK_ID:
      return {get_softmax_bwd_task_impl(), get_softmax_bwd_signature()};
    case CONCAT_FWD_TASK_ID:
      return {get_concat_fwd_task_impl(), get_concat_fwd_signature()};
    case CONCAT_BWD_TASK_ID:
      return {get_concat_bwd_task_impl(), get_concat_bwd_signature()};
    case SPLIT_FWD_TASK_ID:
      return {get_split_fwd_task_impl(), get_split_fwd_signature()};
    case SPLIT_BWD_TASK_ID:
      return {get_split_bwd_task_impl(), get_split_bwd_signature()};
    case REDUCE_INIT_TASK_ID:
      return {get_reduce_init_task_impl(), get_reduce_init_signature()};
    case REDUCE_FWD_TASK_ID:
      return {get_reduce_fwd_task_impl(), get_reduce_fwd_signature()};
    case REDUCE_BWD_TASK_ID:
      return {get_reduce_bwd_task_impl(), get_reduce_bwd_signature()};
    case RESHAPE_INIT_TASK_ID:
      return {get_reshape_init_task_impl(), get_reshape_init_signature()};
    case RESHAPE_FWD_TASK_ID:
      return {get_reshape_fwd_task_impl(), get_reshape_fwd_signature()};
    case RESHAPE_BWD_TASK_ID:
      return {get_reshape_bwd_task_impl(), get_reshape_bwd_signature()};
    case REVERSE_FWD_TASK_ID:
      return {get_reverse_fwd_task_impl(), get_reverse_fwd_signature()};
    case REVERSE_BWD_TASK_ID:
      return {get_reverse_bwd_task_impl(), get_reverse_bwd_signature()};
    case TOPK_INIT_TASK_ID:
      return {get_topk_init_task_impl(), get_topk_init_signature()};
    case TOPK_FWD_TASK_ID:
      return {get_topk_fwd_task_impl(), get_topk_fwd_signature()};
    case TOPK_BWD_TASK_ID:
      return {get_topk_bwd_task_impl(), get_topk_bwd_signature()};
    case TRANSPOSE_INIT_TASK_ID:
      return {get_transpose_init_task_impl(), get_transpose_init_signature()};
    case TRANSPOSE_FWD_TASK_ID:
      return {get_transpose_fwd_task_impl(), get_transpose_fwd_signature()};
    case TRANSPOSE_BWD_TASK_ID:
      return {get_transpose_bwd_task_impl(), get_transpose_bwd_signature()};
    case ATTENTION_INIT_TASK_ID:
      return {get_attention_init_task_impl(), get_attention_init_signature()};
    case ATTENTION_FWD_TASK_ID:
      return {get_attention_fwd_task_impl(), get_attention_fwd_signature()};
    case ATTENTION_BWD_TASK_ID:
      return {get_attention_bwd_task_impl(), get_attention_bwd_signature()};
    case COMBINE_FWD_TASK_ID:
      return {get_combine_fwd_task_impl(), get_combine_fwd_signature()};
    case COMBINE_BWD_TASK_ID:
      return {get_combine_bwd_task_impl(), get_combine_bwd_signature()};
    case REDUCTION_FWD_TASK_ID:
      return {get_reduction_fwd_task_impl(), get_reduction_fwd_signature()};
    case REDUCTION_BWD_TASK_ID:
      return {get_reduction_bwd_task_impl(), get_reduction_bwd_signature()};
    case REPARTITION_INIT_TASK_ID:
      return {get_repartition_init_task_impl(),
              get_repartition_init_signature()};
    case REPARTITION_FWD_TASK_ID:
      return {get_repartition_fwd_task_impl(), get_repartition_fwd_signature()};
    case REPARTITION_BWD_TASK_ID:
      return {get_repartition_bwd_task_impl(), get_repartition_bwd_signature()};
    case REPLICATE_FWD_TASK_ID:
      return {get_replicate_fwd_task_impl(), get_replicate_fwd_signature()};
    case REPLICATE_BWD_TASK_ID:
      return {get_replicate_bwd_task_impl(), get_replicate_bwd_signature()};
    default:
      throw mk_runtime_error(
          fmt::format("Invalid task ID")); // inserting task_id yields
                                           // "type_is_unformattable" error
  }
}
} // namespace FlexFlow
