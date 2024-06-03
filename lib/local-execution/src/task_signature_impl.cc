#include "local-execution/task_signature_impl.h"
#include "ops/element_binary.h"

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
      return {get_elementunary_init_task_impl(),
              init_signature<ELEMENTUNARY_INIT_TASK_ID>()};
    case ELEMENTUNARY_FWD_TASK_ID:
      return {get_elementunary_fwd_task_impl(),
              fwd_signature<ELEMENTUNARY_FWD_TASK_ID>()};
    case ELEMENTUNARY_BWD_TASK_ID:
      return {get_elementunary_bwd_task_impl(),
              bwd_signature<ELEMENTUNARY_BWD_TASK_ID>()};
    case CONV2D_INIT_TASK_ID:
      return {get_conv2d_init_task_impl(),
              init_signature<CONV2D_INIT_TASK_ID>()};
    case CONV2D_FWD_TASK_ID:
      return {get_conv2d_fwd_task_impl(), fwd_signature<CONV2D_FWD_TASK_ID>()};
    case CONV2D_BWD_TASK_ID:
      return {get_conv2d_bwd_task_impl(), bwd_signature<CONV2D_BWD_TASK_ID>()};
    case DROPOUT_INIT_TASK_ID:
      return {get_dropout_init_task_impl(),
              init_signature<DROPOUT_INIT_TASK_ID>()};
    case DROPOUT_FWD_TASK_ID:
      return {get_dropout_fwd_task_impl(),
              fwd_signature<DROPOUT_FWD_TASK_ID>()};
    case DROPOUT_BWD_TASK_ID:
      return {get_dropout_bwd_task_impl(),
              bwd_signature<DROPOUT_BWD_TASK_ID>()};
    case EMBED_INIT_TASK_ID:
      return {get_embed_init_task_impl(), init_signature<EMBED_INIT_TASK_ID>()};
    case EMBED_FWD_TASK_ID:
      return {get_embed_fwd_task_impl(), fwd_signature<EMBED_FWD_TASK_ID>()};
    case EMBED_BWD_TASK_ID:
      return {get_embed_bwd_task_impl(), bwd_signature<EMBED_BWD_TASK_ID>()};
    case GATHER_INIT_TASK_ID:
      return {get_gather_init_task_impl(),
              init_signature<GATHER_INIT_TASK_ID>()};
    case GATHER_FWD_TASK_ID:
      return {get_gather_fwd_task_impl(), fwd_signature<GATHER_FWD_TASK_ID>()};
    case GATHER_BWD_TASK_ID:
      return {get_gather_bwd_task_impl(), bwd_signature<GATHER_BWD_TASK_ID>()};
    case CAST_INIT_TASK_ID:
      return {get_cast_init_task_impl(), init_signature<CAST_INIT_TASK_ID>()};
    case CAST_FWD_TASK_ID:
      return {get_cast_fwd_task_impl(), fwd_signature<CAST_FWD_TASK_ID>()};
    case CAST_BWD_TASK_ID:
      return {get_cast_bwd_task_impl(), bwd_signature<CAST_BWD_TASK_ID>()};
    case POOL2D_INIT_TASK_ID:
      return {get_pool2d_init_task_impl(),
              init_signature<POOL2D_INIT_TASK_ID>()};
    case POOL2D_FWD_TASK_ID:
      return {get_pool2d_fwd_task_impl(), fwd_signature<POOL2D_FWD_TASK_ID>()};
    case POOL2D_BWD_TASK_ID:
      return {get_pool2d_bwd_task_impl(), bwd_signature<POOL2D_BWD_TASK_ID>()};
    case BATCHNORM_INIT_TASK_ID:
      return {get_batchnorm_init_task_impl(),
              init_signature<BATCHNORM_INIT_TASK_ID>()};
    case BATCHNORM_FWD_TASK_ID:
      return {get_batchnorm_fwd_task_impl(),
              fwd_signature<BATCHNORM_FWD_TASK_ID>()};
    case BATCHNORM_BWD_TASK_ID:
      return {get_batchnorm_bwd_task_impl(),
              bwd_signature<BATCHNORM_BWD_TASK_ID>()};
    case BATCHMATMUL_INIT_TASK_ID:
      return {get_batchmatmul_init_task_impl(),
              init_signature<BATCHMATMUL_INIT_TASK_ID>()};
    case BATCHMATMUL_FWD_TASK_ID:
      return {get_batchmatmul_fwd_task_impl(),
              fwd_signature<BATCHMATMUL_FWD_TASK_ID>()};
    case BATCHMATMUL_BWD_TASK_ID:
      return {get_batchmatmul_bwd_task_impl(),
              bwd_signature<BATCHMATMUL_BWD_TASK_ID>()};
    case LAYERNORM_INIT_TASK_ID:
      return {get_layernorm_init_task_impl(),
              init_signature<LAYERNORM_INIT_TASK_ID>()};
    case LAYERNORM_FWD_TASK_ID:
      return {get_layernorm_fwd_task_impl(),
              fwd_signature<LAYERNORM_FWD_TASK_ID>()};
    case LAYERNORM_BWD_TASK_ID:
      return {get_layernorm_bwd_task_impl(),
              bwd_signature<LAYERNORM_BWD_TASK_ID>()};
    case LINEAR_INIT_TASK_ID:
      return {get_linear_init_task_impl(),
              init_signature<LINEAR_INIT_TASK_ID>()};
    case LINEAR_FWD_TASK_ID:
      return {get_linear_fwd_task_impl(), fwd_signature<LINEAR_FWD_TASK_ID>()};
    case LINEAR_BWD_TASK_ID:
      return {get_linear_bwd_task_impl(), bwd_signature<LINEAR_BWD_TASK_ID>()};
    case FLAT_INIT_TASK_ID:
      return {get_flat_init_task_impl(), init_signature<FLAT_INIT_TASK_ID>()};
    case FLAT_FWD_TASK_ID:
      return {get_flat_fwd_task_impl(), fwd_signature<FLAT_FWD_TASK_ID>()};
    case FLAT_BWD_TASK_ID:
      return {get_flat_bwd_task_impl(), bwd_signature<FLAT_BWD_TASK_ID>()};
    case SOFTMAX_INIT_TASK_ID:
      return {get_softmax_init_task_impl(),
              init_signature<SOFTMAX_INIT_TASK_ID>()};
    case SOFTMAX_FWD_TASK_ID:
      return {get_softmax_fwd_task_impl(),
              fwd_signature<SOFTMAX_FWD_TASK_ID>()};
    case SOFTMAX_BWD_TASK_ID:
      return {get_softmax_bwd_task_impl(),
              bwd_signature<SOFTMAX_BWD_TASK_ID>()};
    case CONCAT_INIT_TASK_ID:
      return {get_concat_init_task_impl(),
              init_signature<CONCAT_INIT_TASK_ID>()};
    case CONCAT_FWD_TASK_ID:
      return {get_concat_fwd_task_impl(), fwd_signature<CONCAT_FWD_TASK_ID>()};
    case CONCAT_BWD_TASK_ID:
      return {get_concat_bwd_task_impl(), bwd_signature<CONCAT_BWD_TASK_ID>()};
    case SPLIT_INIT_TASK_ID:
      return {get_split_init_task_impl(), init_signature<SPLIT_INIT_TASK_ID>()};
    case SPLIT_FWD_TASK_ID:
      return {get_split_fwd_task_impl(), fwd_signature<SPLIT_FWD_TASK_ID>()};
    case SPLIT_BWD_TASK_ID:
      return {get_split_bwd_task_impl(), bwd_signature<SPLIT_BWD_TASK_ID>()};
    case REDUCE_INIT_TASK_ID:
      return {get_reduce_init_task_impl(),
              init_signature<REDUCE_INIT_TASK_ID>()};
    case REDUCE_FWD_TASK_ID:
      return {get_reduce_fwd_task_impl(), fwd_signature<REDUCE_FWD_TASK_ID>()};
    case REDUCE_BWD_TASK_ID:
      return {get_reduce_bwd_task_impl(), bwd_signature<REDUCE_BWD_TASK_ID>()};
    case RESHAPE_INIT_TASK_ID:
      return {get_reshape_init_task_impl(),
              init_signature<RESHAPE_INIT_TASK_ID>()};
    case RESHAPE_FWD_TASK_ID:
      return {get_reshape_fwd_task_impl(),
              fwd_signature<RESHAPE_FWD_TASK_ID>()};
    case RESHAPE_BWD_TASK_ID:
      return {get_reshape_bwd_task_impl(),
              bwd_signature<RESHAPE_BWD_TASK_ID>()};
    case REVERSE_INIT_TASK_ID:
      return {get_reverse_init_task_impl(),
              init_signature<REVERSE_INIT_TASK_ID>()};
    case REVERSE_FWD_TASK_ID:
      return {get_reverse_fwd_task_impl(),
              fwd_signature<REVERSE_FWD_TASK_ID>()};
    case REVERSE_BWD_TASK_ID:
      return {get_reverse_bwd_task_impl(),
              bwd_signature<REVERSE_BWD_TASK_ID>()};
    case TOPK_INIT_TASK_ID:
      return {get_topk_init_task_impl(), init_signature<TOPK_INIT_TASK_ID>()};
    case TOPK_FWD_TASK_ID:
      return {get_topk_fwd_task_impl(), fwd_signature<TOPK_FWD_TASK_ID>()};
    case TOPK_BWD_TASK_ID:
      return {get_topk_bwd_task_impl(), bwd_signature<TOPK_BWD_TASK_ID>()};
    case TRANSPOSE_INIT_TASK_ID:
      return {get_transpose_init_task_impl(),
              init_signature<TRANSPOSE_INIT_TASK_ID>()};
    case TRANSPOSE_FWD_TASK_ID:
      return {get_transpose_fwd_task_impl(),
              fwd_signature<TRANSPOSE_FWD_TASK_ID>()};
    case TRANSPOSE_BWD_TASK_ID:
      return {get_transpose_bwd_task_impl(),
              bwd_signature<TRANSPOSE_BWD_TASK_ID>()};
    case ATTENTION_INIT_TASK_ID:
      return {get_attention_init_task_impl(),
              init_signature<ATTENTION_INIT_TASK_ID>()};
    case ATTENTION_FWD_TASK_ID:
      return {get_attention_fwd_task_impl(),
              fwd_signature<ATTENTION_FWD_TASK_ID>()};
    case ATTENTION_BWD_TASK_ID:
      return {get_attention_bwd_task_impl(),
              bwd_signature<ATTENTION_BWD_TASK_ID>()};
    default:
      throw mk_runtime_error(
          fmt::format("Invalid task ID")); // inserting task_id yields
                                           // "type_is_unformattable" error
  }
}
} // namespace FlexFlow
