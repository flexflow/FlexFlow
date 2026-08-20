#include "realm-execution/tasks/task_id_t.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizers/adam_optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "utils/optional.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<task_id_t>
    get_task_id_for_op(DynamicNodeAttrs const &node_attrs,
                       std::optional<OptimizerAttrs> const &optimizer_attrs) {
  DynamicTaskType task_type = assert_unwrap(node_attrs.task_type);
  switch (task_type) {
    case DynamicTaskType::FWD:
      return get_fwd_task_id_for_op_attrs(
          assert_unwrap(node_attrs.op_attrs).require_pcg_op());
    case DynamicTaskType::BWD:
      return get_bwd_task_id_for_op_attrs(
          assert_unwrap(node_attrs.op_attrs).require_pcg_op());
    case DynamicTaskType::UPD:
      return get_update_task_id_for_optimizer_attrs(
          assert_unwrap(optimizer_attrs));
    case DynamicTaskType::LOSS:
      return task_id_t::LOSS_BWD_TASK_ID;
    default:
      PANIC("Unhandled DynamicTaskType", task_type);
  }
}

std::optional<task_id_t>
    get_init_task_id_for_op_attrs(PCGOperatorAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<task_id_t>>(overload{
      [](BatchMatmulAttrs const &) { return std::nullopt; },
      [](BatchNormAttrs const &) { return task_id_t::BATCHNORM_INIT_TASK_ID; },
      [](BroadcastAttrs const &) { return std::nullopt; },
      [](CastAttrs const &) { return std::nullopt; },
      [](CombineAttrs const &) { return std::nullopt; },
      [](ConcatAttrs const &) { return std::nullopt; },
      [](Conv2DAttrs const &) { return task_id_t::CONV2D_INIT_TASK_ID; },
      [](DropoutAttrs const &) { return task_id_t::DROPOUT_INIT_TASK_ID; },
      [](ElementBinaryAttrs const &) {
        return task_id_t::ELEMENTBINARY_INIT_TASK_ID;
      },
      [](ElementUnaryAttrs const &) {
        return task_id_t::ELEMENTUNARY_INIT_TASK_ID;
      },
      [](EmbeddingAttrs const &) { return std::nullopt; },
      [](FlatAttrs const &) { return std::nullopt; },
      [](GatherAttrs const &) { return task_id_t::GATHER_INIT_TASK_ID; },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return task_id_t::LAYERNORM_INIT_TASK_ID; },
      [](LinearAttrs const &) { return task_id_t::LINEAR_INIT_TASK_ID; },
      [](MultiHeadAttentionAttrs const &) {
        return task_id_t::ATTENTION_INIT_TASK_ID;
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return task_id_t::POOL2D_INIT_TASK_ID; },
      [](ReduceAttrs const &) { return task_id_t::REDUCE_INIT_TASK_ID; },
      [](ReductionAttrs const &) { return std::nullopt; },
      [](RepartitionAttrs const &) { return std::nullopt; },
      [](ReplicateAttrs const &) { return std::nullopt; },
      [](ReshapeAttrs const &) { return std::nullopt; },
      [](ReverseAttrs const &) { return std::nullopt; },
      [](SoftmaxAttrs const &) { return task_id_t::SOFTMAX_INIT_TASK_ID; },
      [](SplitAttrs const &) { return std::nullopt; },
      [](TopKAttrs const &) { return std::nullopt; },
      [](TransposeAttrs const &) { return std::nullopt; },
      [](UpsampleAttrs const &) { return std::nullopt; },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<task_id_t>
    get_fwd_task_id_for_op_attrs(PCGOperatorAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<task_id_t>>(overload{
      [](BatchMatmulAttrs const &) {
        return task_id_t::BATCHMATMUL_FWD_TASK_ID;
      },
      [](BatchNormAttrs const &) { return task_id_t::BATCHNORM_FWD_TASK_ID; },
      [](BroadcastAttrs const &) { return task_id_t::BROADCAST_FWD_TASK_ID; },
      [](CastAttrs const &) { return task_id_t::CAST_FWD_TASK_ID; },
      [](CombineAttrs const &) { return std::nullopt; },
      [](ConcatAttrs const &) { return task_id_t::CONCAT_FWD_TASK_ID; },
      [](Conv2DAttrs const &) { return task_id_t::CONV2D_FWD_TASK_ID; },
      [](DropoutAttrs const &) { return task_id_t::DROPOUT_FWD_TASK_ID; },
      [](ElementBinaryAttrs const &) {
        return task_id_t::ELEMENTBINARY_FWD_TASK_ID;
      },
      [](ElementUnaryAttrs const &) {
        return task_id_t::ELEMENTUNARY_FWD_TASK_ID;
      },
      [](EmbeddingAttrs const &) { return task_id_t::EMBED_FWD_TASK_ID; },
      [](FlatAttrs const &) { return task_id_t::FLAT_FWD_TASK_ID; },
      [](GatherAttrs const &) { return task_id_t::GATHER_FWD_TASK_ID; },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return task_id_t::LAYERNORM_FWD_TASK_ID; },
      [](LinearAttrs const &) { return task_id_t::LINEAR_FWD_TASK_ID; },
      [](MultiHeadAttentionAttrs const &) {
        return task_id_t::ATTENTION_FWD_TASK_ID;
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return task_id_t::POOL2D_FWD_TASK_ID; },
      [](ReduceAttrs const &) { return task_id_t::REDUCE_FWD_TASK_ID; },
      [](ReductionAttrs const &) { return std::nullopt; },
      [](RepartitionAttrs const &) { return std::nullopt; },
      [](ReplicateAttrs const &) { return std::nullopt; },
      [](ReshapeAttrs const &) { return task_id_t::RESHAPE_FWD_TASK_ID; },
      [](ReverseAttrs const &) { return task_id_t::REVERSE_FWD_TASK_ID; },
      [](SoftmaxAttrs const &) { return task_id_t::SOFTMAX_FWD_TASK_ID; },
      [](SplitAttrs const &) { return task_id_t::SPLIT_FWD_TASK_ID; },
      [](TopKAttrs const &) { return task_id_t::TOPK_FWD_TASK_ID; },
      [](TransposeAttrs const &) { return task_id_t::TRANSPOSE_FWD_TASK_ID; },
      [](UpsampleAttrs const &) { return task_id_t::UPSAMPLE_FWD_TASK_ID; },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<task_id_t>
    get_bwd_task_id_for_op_attrs(PCGOperatorAttrs const &op_attrs) {

  return op_attrs.visit<std::optional<task_id_t>>(overload{
      [](BatchMatmulAttrs const &) {
        return task_id_t::BATCHMATMUL_BWD_TASK_ID;
      },
      [](BatchNormAttrs const &) { return task_id_t::BATCHNORM_BWD_TASK_ID; },
      [](BroadcastAttrs const &) { return task_id_t::BROADCAST_BWD_TASK_ID; },
      [](CastAttrs const &) { return task_id_t::CAST_BWD_TASK_ID; },
      [](CombineAttrs const &) { return std::nullopt; },
      [](ConcatAttrs const &) { return task_id_t::CONCAT_BWD_TASK_ID; },
      [](Conv2DAttrs const &) { return task_id_t::CONV2D_BWD_TASK_ID; },
      [](DropoutAttrs const &) { return task_id_t::DROPOUT_BWD_TASK_ID; },
      [](ElementBinaryAttrs const &) {
        return task_id_t::ELEMENTBINARY_BWD_TASK_ID;
      },
      [](ElementUnaryAttrs const &) {
        return task_id_t::ELEMENTUNARY_BWD_TASK_ID;
      },
      [](EmbeddingAttrs const &) { return task_id_t::EMBED_BWD_TASK_ID; },
      [](FlatAttrs const &) { return task_id_t::FLAT_BWD_TASK_ID; },
      [](GatherAttrs const &) { return task_id_t::GATHER_BWD_TASK_ID; },
      [](InputAttrs const &) { return std::nullopt; },
      [](LayerNormAttrs const &) { return task_id_t::LAYERNORM_BWD_TASK_ID; },
      [](LinearAttrs const &) { return task_id_t::LINEAR_BWD_TASK_ID; },
      [](MultiHeadAttentionAttrs const &) {
        return task_id_t::ATTENTION_BWD_TASK_ID;
      },
      [](NoopAttrs const &) { return std::nullopt; },
      [](Pool2DAttrs const &) { return task_id_t::POOL2D_BWD_TASK_ID; },
      [](ReduceAttrs const &) { return task_id_t::REDUCE_BWD_TASK_ID; },
      [](ReductionAttrs const &) { return std::nullopt; },
      [](RepartitionAttrs const &) { return std::nullopt; },
      [](ReplicateAttrs const &) { return std::nullopt; },
      [](ReshapeAttrs const &) { return task_id_t::RESHAPE_BWD_TASK_ID; },
      [](ReverseAttrs const &) { return task_id_t::REVERSE_BWD_TASK_ID; },
      [](SoftmaxAttrs const &) { return task_id_t::SOFTMAX_BWD_TASK_ID; },
      [](SplitAttrs const &) { return task_id_t::SPLIT_BWD_TASK_ID; },
      [](TopKAttrs const &) { return task_id_t::TOPK_BWD_TASK_ID; },
      [](TransposeAttrs const &) { return task_id_t::TRANSPOSE_BWD_TASK_ID; },
      [](UpsampleAttrs const &) { return task_id_t::UPSAMPLE_BWD_TASK_ID; },
      [](WeightAttrs const &) { return std::nullopt; },
  });
}

std::optional<task_id_t> get_update_task_id_for_optimizer_attrs(
    OptimizerAttrs const &optimizer_attrs) {

  return optimizer_attrs.visit<std::optional<task_id_t>>(overload{
      [](SGDOptimizerAttrs const &) { return task_id_t::SGD_UPD_NCCL_TASK_ID; },
      [](AdamOptimizerAttrs const &) {
        return task_id_t::ADAM_UPD_NCCL_TASK_ID;
      },
  });
}

Realm::Processor::TaskFuncID get_realm_task_id_for_task_id(task_id_t task_id) {
  return Realm::Processor::TASK_ID_FIRST_AVAILABLE +
         static_cast<Realm::Processor::TaskFuncID>(task_id);
}

} // namespace FlexFlow
