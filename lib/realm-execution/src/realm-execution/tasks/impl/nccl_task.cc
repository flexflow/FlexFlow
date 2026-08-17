#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/nccl_task.h"
#include "realm-execution/tasks/impl/nccl_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "task-spec/permissions.h"
#include "utils/containers/get_only.h"
#include "utils/optional.h"

#include <cstdio>
#include <nccl.h>
#include <stdexcept>

namespace FlexFlow {

ncclResult_t run_nccl_all_reduce(void const *send_buffer,
                                 void *receive_buffer,
                                 size_t count,
                                 ncclDataType_t data_type,
                                 ncclRedOp_t reduction_op,
                                 ncclComm_t communicator,
                                 ffStream_t stream) {
  return ncclAllReduce(send_buffer,
                       receive_buffer,
                       count,
                       data_type,
                       reduction_op,
                       communicator,
                       stream);
}

ncclResult_t run_nccl_broadcast(void const *send_buffer,
                                void *receive_buffer,
                                size_t count,
                                ncclDataType_t data_type,
                                int root_rank,
                                ncclComm_t communicator,
                                ffStream_t stream) {
  return ncclBroadcast(send_buffer,
                       receive_buffer,
                       count,
                       data_type,
                       root_rank,
                       communicator,
                       stream);
}

ncclResult_t run_nccl_reduce(void const *send_buffer,
                             void *receive_buffer,
                             size_t count,
                             ncclDataType_t data_type,
                             ncclRedOp_t reduction_op,
                             int root_rank,
                             ncclComm_t communicator,
                             ffStream_t stream) {
  return ncclReduce(send_buffer,
                    receive_buffer,
                    count,
                    data_type,
                    reduction_op,
                    root_rank,
                    communicator,
                    stream);
}

static ncclDataType_t get_nccl_data_type(DataType data_type) {
    switch (data_type) {
        case DataType::BOOL:
            return ncclUint8;
        case DataType::INT32:
            return ncclInt32;
        case DataType::INT64:
            return ncclInt64;
        case DataType::HALF:
            return ncclHalf;
        case DataType::FLOAT:
            return ncclFloat;
        case DataType::DOUBLE:
            return ncclDouble;
  }

  throw std::runtime_error("Unsupported datatype for NCCL collective");
}

void nccl_task_body(void const *args,
                    size_t arglen,
                    void const *userdata,
                    size_t userdata_len,
                    Realm::Processor proc) {
  (void)userdata;
  (void)userdata_len;

  NCCLTaskArgs task_args = nccl_task_args_from_serializable(
      deserialize_task_args<SerializableNcclTaskArgs>(args, arglen));

  RealmContext ctx{proc};

  global_device_id_t current_device =
      ctx.get_current_global_device_id();

  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_ff_handle(
          task_args.device_handle,
          current_device);

  PerDeviceFFHandle const &gpu_handle =
      device_handle.require_for_gpu();

  ffStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  DynamicNodeInvocation const &invocation =
      task_args.invocation;

  TrainingOperationAttrs const &training_op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  PCGOperatorAttrs const &pcg_op_attrs =
      training_op_attrs.require_pcg_op();

  (void)pcg_op_attrs.require_parallel_replicate();

  DynamicTaskType task_type =
      assert_unwrap(invocation.node_attrs.task_type);

  auto find_local_task_shard =
      [&](auto const &slot_map) -> DynamicValueAttrs const & {
    DynamicValueAttrs const *local_value = nullptr;

    for (auto const &[slot, value] : slot_map) {
      if (slot.task_shard.has_value() &&
          slot.task_shard.value() == current_device.coord) {
        ASSERT(local_value == nullptr);
        local_value = &value;
      }
    }

    ASSERT(local_value != nullptr);
    return *local_value;
  };

  auto get_accessor =
      [&](DynamicValueAttrs const &value,
          Permissions permissions) -> DynamicTensorAccessor {
    auto const &[inst, event] =
        task_args.tensor_backing.backing.at(value);

    return dynamic_tensor_accessor_from_instance(
        inst,
        event,
        assert_unwrap(value.parallel_tensor_shape),
        permissions,
        ctx.get_current_processor());
  };

  ncclResult_t result = ncclSuccess;

  switch (task_type) {
    case DynamicTaskType::FWD: {
      // Replicate forward = NCCL broadcast

      DynamicValueAttrs const &root_input_value =
          get_only(invocation.inputs).second;

      DynamicValueAttrs const &local_output_value =
          find_local_task_shard(invocation.outputs);

      global_device_id_t root_device =
          pt_mapping_get_device_for_coord(
              assert_unwrap(root_input_value.mapping),
              assert_unwrap(root_input_value.shard_coord));
      Realm::Processor root_proc =
          ctx.processor_from_global_device_id(root_device);

      int root_rank =
          static_cast<int>(root_proc.address_space());

      DynamicTensorAccessor local_output_accessor =
          get_accessor(local_output_value, Permissions::RW);

      GenericTensorAccessorW const &local_output =
          local_output_accessor.require_write();

      size_t count =
          get_num_elements(local_output.shape.dims)
              .int_from_positive_int();

      ncclDataType_t nccl_data_type =
          get_nccl_data_type(local_output.shape.data_type);

      void const *send_buffer = local_output.ptr;

      if (current_device == root_device) {
        DynamicTensorAccessor root_input_accessor =
            get_accessor(root_input_value, Permissions::RO);

        GenericTensorAccessorR const &root_input =
            root_input_accessor.require_read();

        ASSERT(root_input.shape == local_output.shape);

        send_buffer = root_input.ptr;
      }

      result = run_nccl_broadcast(
          send_buffer,
          local_output.ptr,
          count,
          nccl_data_type,
          root_rank,
          gpu_handle.ncclComm,
          stream);

      break;
    }

    case DynamicTaskType::BWD: {
      // Replicate backward = NCCL reduce

      DynamicValueAttrs const &local_input_value =
          find_local_task_shard(invocation.inputs);

      DynamicValueAttrs const &root_output_value =
          get_only(invocation.outputs).second;

      global_device_id_t root_device =
          pt_mapping_get_device_for_coord(
              assert_unwrap(root_output_value.mapping),
              assert_unwrap(root_output_value.shard_coord));

      Realm::Processor root_proc =
          ctx.processor_from_global_device_id(root_device);

      int root_rank =
          static_cast<int>(root_proc.address_space());

      DynamicTensorAccessor local_input_accessor =
          get_accessor(local_input_value, Permissions::RO);

      GenericTensorAccessorR const &local_input =
          local_input_accessor.require_read();

      size_t count =
          get_num_elements(local_input.shape.dims)
              .int_from_positive_int();

      ncclDataType_t nccl_data_type =
          get_nccl_data_type(local_input.shape.data_type);

      void *receive_buffer =
          const_cast<void *>(local_input.ptr);

      if (current_device == root_device) {
        DynamicTensorAccessor root_output_accessor =
            get_accessor(root_output_value, Permissions::RW);

        GenericTensorAccessorW const &root_output =
            root_output_accessor.require_write();

        ASSERT(local_input.shape == root_output.shape);

        receive_buffer = root_output.ptr;
      }

      result = run_nccl_reduce(
          local_input.ptr,
          receive_buffer,
          count,
          nccl_data_type,
          ncclSum,
          root_rank,
          gpu_handle.ncclComm,
          stream);

      break;
    }

    default:
      PANIC("Unhandled replicate task type ", task_type);
  }

  if (result != ncclSuccess) {
    std::fprintf(stderr,
                 "NCCL collective failed: %s\n",
                 ncclGetErrorString(result));
    return;
  }
}

Realm::Event spawn_nccl_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    Realm::Event precondition) {
  NCCLTaskArgs task_args = NCCLTaskArgs{
      /*invocation=*/invocation,
      /*tensor_backing=*/tensor_backing,
      /*device_handle=*/device_handle,
  };

  std::string serialized_args =
      serialize_task_args(nccl_task_args_to_serializable(task_args));

  return ctx.spawn_task(
      target_proc,
      task_id_t::NCCL_HELLO_WORLD_TASK_ID,
      serialized_args.data(),
      serialized_args.size(),
      Realm::ProfilingRequestSet{},
      precondition);
}

} // namespace FlexFlow
