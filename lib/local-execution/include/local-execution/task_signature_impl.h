#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H

#include "task_argument_accessor.h"
#include "local-execution/device_specific.h"
#include "local-execution/device_states.h"
#include "local-execution/tasks.h"
#include "utils/variant.h"

namespace FlexFlow {

using TaskImplFunction = std::variant<
    std::function<DeviceSpecific<DeviceStates>(TaskArgumentAccessor const &)>,
    std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

TaskSignatureImpl get_task_sig_impl(task_id_t const &);

TaskImplFunction get_elementbinary_init_task_impl();
TaskImplFunction get_elementbinary_fwd_task_impl();
TaskImplFunction get_elementbinary_bwd_task_impl();
TaskImplFunction get_elementunary_init_task_impl();
TaskImplFunction get_elementunary_fwd_task_impl();
TaskImplFunction get_elementunary_bwd_task_impl();
TaskImplFunction get_conv2d_init_task_impl();
TaskImplFunction get_conv2d_fwd_task_impl();
TaskImplFunction get_conv2d_bwd_task_impl();
TaskImplFunction get_dropout_init_task_impl();
TaskImplFunction get_dropout_fwd_task_impl();
TaskImplFunction get_dropout_bwd_task_impl();
TaskImplFunction get_embed_init_task_impl();
TaskImplFunction get_embed_fwd_task_impl();
TaskImplFunction get_embed_bwd_task_impl();
TaskImplFunction get_gather_init_task_impl();
TaskImplFunction get_gather_fwd_task_impl();
TaskImplFunction get_gather_bwd_task_impl();
TaskImplFunction get_cast_init_task_impl();
TaskImplFunction get_cast_fwd_task_impl();
TaskImplFunction get_cast_bwd_task_impl();
TaskImplFunction get_pool2d_init_task_impl();
TaskImplFunction get_pool2d_fwd_task_impl();
TaskImplFunction get_pool2d_bwd_task_impl();
TaskImplFunction get_batchnorm_init_task_impl();
TaskImplFunction get_batchnorm_fwd_task_impl();
TaskImplFunction get_batchnorm_bwd_task_impl();
TaskImplFunction get_batchmatmul_init_task_impl();
TaskImplFunction get_batchmatmul_fwd_task_impl();
TaskImplFunction get_batchmatmul_bwd_task_impl();
TaskImplFunction get_layernorm_init_task_impl();
TaskImplFunction get_layernorm_fwd_task_impl();
TaskImplFunction get_layernorm_bwd_task_impl();
TaskImplFunction get_linear_init_task_impl();
TaskImplFunction get_linear_fwd_task_impl();
TaskImplFunction get_linear_bwd_task_impl();
TaskImplFunction get_flat_init_task_impl();
TaskImplFunction get_flat_fwd_task_impl();
TaskImplFunction get_flat_bwd_task_impl();
TaskImplFunction get_softmax_init_task_impl();
TaskImplFunction get_softmax_fwd_task_impl();
TaskImplFunction get_softmax_bwd_task_impl();
TaskImplFunction get_concat_init_task_impl();
TaskImplFunction get_concat_fwd_task_impl();
TaskImplFunction get_concat_bwd_task_impl();
TaskImplFunction get_split_init_task_impl();
TaskImplFunction get_split_fwd_task_impl();
TaskImplFunction get_split_bwd_task_impl();
TaskImplFunction get_reduce_init_task_impl();
TaskImplFunction get_reduce_fwd_task_impl();
TaskImplFunction get_reduce_bwd_task_impl();
TaskImplFunction get_reshape_init_task_impl();
TaskImplFunction get_reshape_fwd_task_impl();
TaskImplFunction get_reshape_bwd_task_impl();
TaskImplFunction get_reverse_init_task_impl();
TaskImplFunction get_reverse_fwd_task_impl();
TaskImplFunction get_reverse_bwd_task_impl();
TaskImplFunction get_topk_init_task_impl();
TaskImplFunction get_topk_fwd_task_impl();
TaskImplFunction get_topk_bwd_task_impl();
TaskImplFunction get_transpose_init_task_impl();
TaskImplFunction get_transpose_fwd_task_impl();
TaskImplFunction get_transpose_bwd_task_impl();
TaskImplFunction get_attention_init_task_impl();
TaskImplFunction get_attention_fwd_task_impl();
TaskImplFunction get_attention_bwd_task_impl();

} // namespace FlexFlow

#endif
