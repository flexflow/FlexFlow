#include "task-spec/ops/impl/embedding.h"
#include "kernels/embedding_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Embedding;

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  EmbeddingAttrs attrs = acc.get_op_attrs().require_embedding();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  return profile(
      forward_kernel,
      profiling,
      kernel_device_type,
      "[Embedding] forward_time = {:.2lf}ms\n",
      input,
      output,
      weight,
      input.shape.data_type,
      output.shape.data_type,
      attrs.aggr,
      get_num_dims(input.shape.dims),
      get_num_dims(output.shape.dims),
      dim_at_idx(input.shape.dims, legion_dim_t{1_n}).int_from_positive_int());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto weight_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::WEIGHT);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  EmbeddingAttrs attrs = acc.get_op_attrs().require_embedding();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  return profile(
      backward_kernel,
      profiling,
      kernel_device_type,
      "[Embedding] backward_time = {:.2lf}ms\n",
      output,
      input,
      weight_grad,
      output.shape.data_type,
      input.shape.data_type,
      attrs.aggr,
      get_num_dims(input.shape.dims),
      get_num_dims(output.shape.dims),
      dim_at_idx(input.shape.dims, ff_dim_t{0_n}).int_from_positive_int());
}

TaskImplFunction get_embedding_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_embedding_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow
