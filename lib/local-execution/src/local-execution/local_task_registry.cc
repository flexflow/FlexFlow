#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "task-spec/loss_functions.h"
#include "task-spec/ops/impl/attention.h"
#include "task-spec/ops/impl/batch_matmul.h"
#include "task-spec/ops/impl/batch_norm.h"
#include "task-spec/ops/impl/broadcast.h"
#include "task-spec/ops/impl/cast.h"
#include "task-spec/ops/impl/concat.h"
#include "task-spec/ops/impl/conv_2d.h"
#include "task-spec/ops/impl/dropout.h"
#include "task-spec/ops/impl/element_binary.h"
#include "task-spec/ops/impl/element_unary.h"
#include "task-spec/ops/impl/embedding.h"
#include "task-spec/ops/impl/flat.h"
#include "task-spec/ops/impl/gather.h"
#include "task-spec/ops/impl/layer_norm.h"
#include "task-spec/ops/impl/linear.h"
#include "task-spec/ops/impl/pool_2d.h"
#include "task-spec/ops/impl/reduce.h"
#include "task-spec/ops/impl/reshape.h"
#include "task-spec/ops/impl/reverse.h"
#include "task-spec/ops/impl/softmax.h"
#include "task-spec/ops/impl/split.h"
#include "task-spec/ops/impl/topk.h"
#include "task-spec/ops/impl/transpose.h"
#include "task-spec/ops/impl/upsample.h"
#include "task-spec/optimizer.h"
#include "task-spec/task_impl_function.dtg.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include <optional>

namespace FlexFlow {

std::optional<TaskImplFunction>
    get_init_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<TaskImplFunction>>(overload{
      [](BatchMatmulAttrs const &) { return std::nullopt; },
      [](BatchNormAttrs const &) { return get_batch_norm_init_task_impl(); },
      [](BroadcastAttrs const &) { return std::nullopt; },
      [](CastAttrs const &) { return std::nullopt; },
      [](ConcatAttrs const &) { return std::nullopt; },
      [](Conv2DAttrs const &) { return get_conv_2d_init_task_impl(); },
      [](DropoutAttrs const &) { return get_dropout_init_task_impl(); },
      [](ElementBinaryAttrs const &) {
        return get_element_binary_init_task_impl();
      },
      [](ElementUnaryAttrs const &) {
        return get_element_unary_init_task_impl();
      },
      [](EmbeddingAttrs const &) { return std::nullopt; },
      [](FlatAttrs const &) { return std::nullopt; },
      [](GatherAttrs const &) { return get_gather_init_task_impl(); },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return get_layer_norm_init_task_impl(); },
      [](LinearAttrs const &) { return get_linear_init_task_impl(); },
      [](MultiHeadAttentionAttrs const &) {
        return get_attention_init_task_impl();
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return get_pool_2d_init_task_impl(); },
      [](ReduceAttrs const &) { return get_reduce_init_task_impl(); },
      [](ReshapeAttrs const &) { return std::nullopt; },
      [](ReverseAttrs const &) { return std::nullopt; },
      [](SoftmaxAttrs const &) { return get_softmax_init_task_impl(); },
      [](SplitAttrs const &) { return std::nullopt; },
      [](TopKAttrs const &) { return std::nullopt; },
      [](TransposeAttrs const &) { return std::nullopt; },
      [](UpsampleAttrs const &) { return std::nullopt; },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<TaskImplFunction>
    get_fwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<TaskImplFunction>>(overload{
      [](BatchMatmulAttrs const &) { return get_batch_matmul_fwd_task_impl(); },
      [](BatchNormAttrs const &) { return get_batch_norm_fwd_task_impl(); },
      [](BroadcastAttrs const &) { return get_broadcast_fwd_task_impl(); },
      [](CastAttrs const &) { return get_cast_fwd_task_impl(); },
      [](ConcatAttrs const &) { return get_concat_fwd_task_impl(); },
      [](Conv2DAttrs const &) { return get_conv_2d_fwd_task_impl(); },
      [](DropoutAttrs const &) { return get_dropout_fwd_task_impl(); },
      [](ElementBinaryAttrs const &) {
        return get_element_binary_fwd_task_impl();
      },
      [](ElementUnaryAttrs const &) {
        return get_element_unary_fwd_task_impl();
      },
      [](EmbeddingAttrs const &) { return get_embedding_fwd_task_impl(); },
      [](FlatAttrs const &) { return get_flat_fwd_task_impl(); },
      [](GatherAttrs const &) { return get_gather_fwd_task_impl(); },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return get_layer_norm_fwd_task_impl(); },
      [](LinearAttrs const &) { return get_linear_fwd_task_impl(); },
      [](MultiHeadAttentionAttrs const &) {
        return get_attention_fwd_task_impl();
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return get_pool_2d_fwd_task_impl(); },
      [](ReduceAttrs const &) { return get_reduce_fwd_task_impl(); },
      [](ReshapeAttrs const &) { return get_reshape_fwd_task_impl(); },
      [](ReverseAttrs const &) { return get_reverse_fwd_task_impl(); },
      [](SoftmaxAttrs const &) { return get_softmax_fwd_task_impl(); },
      [](SplitAttrs const &) { return get_split_fwd_task_impl(); },
      [](TopKAttrs const &) { return get_topk_fwd_task_impl(); },
      [](TransposeAttrs const &) { return get_transpose_fwd_task_impl(); },
      [](UpsampleAttrs const &) { return get_upsample_fwd_task_impl(); },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<TaskImplFunction>
    get_bwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<TaskImplFunction>>(overload{
      [](BatchMatmulAttrs const &) { return get_batch_matmul_bwd_task_impl(); },
      [](BatchNormAttrs const &) { return get_batch_norm_bwd_task_impl(); },
      [](BroadcastAttrs const &) { return get_broadcast_bwd_task_impl(); },
      [](CastAttrs const &) { return get_cast_bwd_task_impl(); },
      [](ConcatAttrs const &) { return get_concat_bwd_task_impl(); },
      [](Conv2DAttrs const &) { return get_conv_2d_bwd_task_impl(); },
      [](DropoutAttrs const &) { return get_dropout_bwd_task_impl(); },
      [](ElementBinaryAttrs const &) {
        return get_element_binary_bwd_task_impl();
      },
      [](ElementUnaryAttrs const &) {
        return get_element_unary_bwd_task_impl();
      },
      [](EmbeddingAttrs const &) { return get_embedding_bwd_task_impl(); },
      [](FlatAttrs const &) { return get_flat_bwd_task_impl(); },
      [](GatherAttrs const &) { return get_gather_bwd_task_impl(); },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return get_layer_norm_bwd_task_impl(); },
      [](LinearAttrs const &) { return get_linear_bwd_task_impl(); },
      [](MultiHeadAttentionAttrs const &) {
        return get_attention_bwd_task_impl();
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return get_pool_2d_bwd_task_impl(); },
      [](ReduceAttrs const &) { return get_reduce_bwd_task_impl(); },
      [](ReshapeAttrs const &) { return get_reshape_bwd_task_impl(); },
      [](ReverseAttrs const &) { return get_reverse_bwd_task_impl(); },
      [](SoftmaxAttrs const &) { return get_softmax_bwd_task_impl(); },
      [](SplitAttrs const &) { return get_split_bwd_task_impl(); },
      [](TopKAttrs const &) { return get_topk_bwd_task_impl(); },
      [](TransposeAttrs const &) { return get_transpose_bwd_task_impl(); },
      [](UpsampleAttrs const &) { return get_upsample_bwd_task_impl(); },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<DeviceSpecificPerDeviceOpState>
    call_init_task_impl(ComputationGraphOpAttrs const &op_attrs,
                        TaskArgumentAccessor const &arg_accessor) {
  std::optional<TaskImplFunction> task_impl_fn =
      get_init_task_impl_for_op_attrs(op_attrs);
  if (!task_impl_fn.has_value()) {
    return std::nullopt;
  }

  auto fn =
      assert_unwrap(task_impl_fn).get<InitOpTaskImplFunction>().function_ptr;

  std::optional<DeviceSpecificPerDeviceOpState> device_state = fn(arg_accessor);

  return device_state;
}

std::optional<milliseconds_t>
    call_fwd_task_impl(ComputationGraphOpAttrs const &op_attrs,
                       TaskArgumentAccessor const &acc) {
  std::optional<TaskImplFunction> task_impl_fn =
      get_fwd_task_impl_for_op_attrs(op_attrs);
  if (!task_impl_fn.has_value()) {
    return std::nullopt;
  }
  auto fn =
      assert_unwrap(task_impl_fn).get<FwdBwdOpTaskImplFunction>().function_ptr;

  return fn(acc);
}

std::optional<milliseconds_t>
    call_bwd_task_impl(ComputationGraphOpAttrs const &op_attrs,
                       TaskArgumentAccessor const &acc) {
  std::optional<TaskImplFunction> task_impl_fn =
      get_bwd_task_impl_for_op_attrs(op_attrs);
  if (!task_impl_fn.has_value()) {
    return std::nullopt;
  }
  auto fn =
      assert_unwrap(task_impl_fn).get<FwdBwdOpTaskImplFunction>().function_ptr;

  return fn(acc);
}

void call_update_task_impl(OptimizerAttrs const &optimizer_attrs,
                           TaskArgumentAccessor const &acc) {
  TaskImplFunction task_impl_fn = get_update_task_impl(optimizer_attrs);
  auto fn = task_impl_fn.get<GenericTaskImplFunction>().function_ptr;

  return fn(acc);
}

void call_loss_task_impl(TaskArgumentAccessor const &acc) {
  TaskImplFunction task_impl_fn = get_loss_bwd_task_impl();
  auto fn = task_impl_fn.get<GenericTaskImplFunction>().function_ptr;

  return fn(acc);
}

} // namespace FlexFlow
