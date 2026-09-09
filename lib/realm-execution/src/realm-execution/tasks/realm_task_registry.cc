#include "realm-execution/tasks/realm_task_registry.h"
#include "realm-execution/tasks/impl/controller_task.h"
#include "realm-execution/tasks/impl/ff_handle_init_return_task.h"
#include "realm-execution/tasks/impl/ff_handle_init_task.h"
#include "realm-execution/tasks/impl/op_task.h"
#include "realm-execution/tasks/impl/per_device_op_state_init_return_task.h"
#include "realm-execution/tasks/impl/per_device_op_state_init_task.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/exception.h"

namespace FlexFlow {

Realm::Event register_task(Realm::Processor::Kind target_kind,
                           task_id_t func_id,
                           void (*task_body)(void const *,
                                             size_t,
                                             void const *,
                                             size_t,
                                             Realm::Processor)) {
  Realm::Processor::TaskFuncID realm_task_id =
      get_realm_task_id_for_task_id(func_id);
#ifdef FF_USE_PREALM
  Realm::prealm_task_name(realm_task_id, fmt::to_string(func_id));
#endif
  return Realm::Processor::register_task_by_kind(
      target_kind,
      /*global=*/false,
      realm_task_id,
      Realm::CodeDescriptor(task_body),
      Realm::ProfilingRequestSet());
}

Realm::Event register_all_tasks() {
  std::vector<Realm::Event> pending_registrations;

  std::vector<task_id_t> init_task_ids = {
      // Init tasks
      task_id_t::BATCHNORM_INIT_TASK_ID,
      task_id_t::COMBINE_INIT_TASK_ID,
      task_id_t::CONV2D_INIT_TASK_ID,
      task_id_t::DROPOUT_INIT_TASK_ID,
      task_id_t::ELEMENTBINARY_INIT_TASK_ID,
      task_id_t::ELEMENTUNARY_INIT_TASK_ID,
      task_id_t::GATHER_INIT_TASK_ID,
      task_id_t::LAYERNORM_INIT_TASK_ID,
      task_id_t::LINEAR_INIT_TASK_ID,
      task_id_t::ATTENTION_INIT_TASK_ID,
      task_id_t::POOL2D_INIT_TASK_ID,
      task_id_t::REDUCE_INIT_TASK_ID,
      task_id_t::REDUCTION_INIT_TASK_ID,
      task_id_t::REPARTITION_INIT_TASK_ID,
      task_id_t::SOFTMAX_INIT_TASK_ID,
  };

  for (task_id_t task_id : init_task_ids) {
    pending_registrations.push_back(
        register_task(Realm::Processor::LOC_PROC,
                      task_id,
                      per_device_op_state_init_task_body));
    pending_registrations.push_back(
        register_task(Realm::Processor::TOC_PROC,
                      task_id,
                      per_device_op_state_init_task_body));
  }

  std::vector<task_id_t> task_ids = {
      // Forward tasks
      task_id_t::BATCHMATMUL_FWD_TASK_ID,
      task_id_t::BATCHNORM_FWD_TASK_ID,
      task_id_t::BROADCAST_FWD_TASK_ID,
      task_id_t::CAST_FWD_TASK_ID,
      task_id_t::COMBINE_FWD_TASK_ID,
      task_id_t::CONCAT_FWD_TASK_ID,
      task_id_t::CONV2D_FWD_TASK_ID,
      task_id_t::DROPOUT_FWD_TASK_ID,
      task_id_t::ELEMENTBINARY_FWD_TASK_ID,
      task_id_t::ELEMENTUNARY_FWD_TASK_ID,
      task_id_t::EMBED_FWD_TASK_ID,
      task_id_t::FLAT_FWD_TASK_ID,
      task_id_t::GATHER_FWD_TASK_ID,
      task_id_t::LAYERNORM_FWD_TASK_ID,
      task_id_t::LINEAR_FWD_TASK_ID,
      task_id_t::ATTENTION_FWD_TASK_ID,
      task_id_t::POOL2D_FWD_TASK_ID,
      task_id_t::REDUCE_FWD_TASK_ID,
      task_id_t::REDUCTION_FWD_TASK_ID,
      task_id_t::REPARTITION_FWD_TASK_ID,
      task_id_t::RESHAPE_FWD_TASK_ID,
      task_id_t::REVERSE_FWD_TASK_ID,
      task_id_t::SOFTMAX_FWD_TASK_ID,
      task_id_t::SPLIT_FWD_TASK_ID,
      task_id_t::TOPK_FWD_TASK_ID,
      task_id_t::TRANSPOSE_FWD_TASK_ID,
      task_id_t::UPSAMPLE_FWD_TASK_ID,

      // Backward tasks
      task_id_t::BATCHMATMUL_BWD_TASK_ID,
      task_id_t::BATCHNORM_BWD_TASK_ID,
      task_id_t::BROADCAST_BWD_TASK_ID,
      task_id_t::CAST_BWD_TASK_ID,
      task_id_t::COMBINE_BWD_TASK_ID,
      task_id_t::CONCAT_BWD_TASK_ID,
      task_id_t::CONV2D_BWD_TASK_ID,
      task_id_t::DROPOUT_BWD_TASK_ID,
      task_id_t::ELEMENTBINARY_BWD_TASK_ID,
      task_id_t::ELEMENTUNARY_BWD_TASK_ID,
      task_id_t::EMBED_BWD_TASK_ID,
      task_id_t::FLAT_BWD_TASK_ID,
      task_id_t::GATHER_BWD_TASK_ID,
      task_id_t::LAYERNORM_BWD_TASK_ID,
      task_id_t::LINEAR_BWD_TASK_ID,
      task_id_t::ATTENTION_BWD_TASK_ID,
      task_id_t::POOL2D_BWD_TASK_ID,
      task_id_t::REDUCE_BWD_TASK_ID,
      task_id_t::REDUCTION_BWD_TASK_ID,
      task_id_t::REPARTITION_BWD_TASK_ID,
      task_id_t::RESHAPE_BWD_TASK_ID,
      task_id_t::REVERSE_BWD_TASK_ID,
      task_id_t::SOFTMAX_BWD_TASK_ID,
      task_id_t::SPLIT_BWD_TASK_ID,
      task_id_t::TOPK_BWD_TASK_ID,
      task_id_t::TRANSPOSE_BWD_TASK_ID,
      task_id_t::UPSAMPLE_BWD_TASK_ID,

      // Update tasks
      task_id_t::SGD_UPD_NCCL_TASK_ID,
      task_id_t::ADAM_UPD_NCCL_TASK_ID,

      // Loss task
      task_id_t::LOSS_BWD_TASK_ID,
  };

  for (task_id_t task_id : task_ids) {
    pending_registrations.push_back(
        register_task(Realm::Processor::LOC_PROC, task_id, op_task_body));
    pending_registrations.push_back(
        register_task(Realm::Processor::TOC_PROC, task_id, op_task_body));
  }

  pending_registrations.push_back(register_task(Realm::Processor::LOC_PROC,
                                                task_id_t::CONTROLLER_TASK_ID,
                                                controller_task_body));
  pending_registrations.push_back(
      register_task(Realm::Processor::LOC_PROC,
                    task_id_t::DEVICE_HANDLE_INIT_TASK_ID,
                    ff_handle_init_task_body));
  pending_registrations.push_back(
      register_task(Realm::Processor::TOC_PROC,
                    task_id_t::DEVICE_HANDLE_INIT_TASK_ID,
                    ff_handle_init_task_body));
  pending_registrations.push_back(
      register_task(Realm::Processor::LOC_PROC,
                    task_id_t::DEVICE_HANDLE_INIT_RETURN_TASK_ID,
                    ff_handle_init_return_task_body));
  pending_registrations.push_back(
      register_task(Realm::Processor::LOC_PROC,
                    task_id_t::DEVICE_STATE_INIT_RETURN_TASK_ID,
                    per_device_op_state_init_return_task_body));
  return Realm::Event::merge_events(pending_registrations);
}

} // namespace FlexFlow
