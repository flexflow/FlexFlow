#include "local-execution/optimizer.h"
#include "kernels/optimizer_kernels.h"
#include "local-execution/profiling.h"
#include "utils/overload.h"

namespace FlexFlow {

enum Slots { ATTRS, WEIGHT, SGD_V, PROFILING, ADAM_M, ADAM_V, HANDLE };

TaskSignature get_sgd_update_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, WEIGHT, IsGrad::YES);
  add_slot(sig, WEIGHT, IsGrad::NO);
  add_slot(sig, SGD_V, IsGrad::YES);
  add_arg_slot<SGDOptimizerAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    add_unchecked_arg_slot<PerDeviceFFHandle>(sig, HANDLE);
  }
  return sig;
}

TaskInvocation sgd_update(SGDOptimizerAttrs const &attrs,
                          tensor_guid_t const &weight,
                          tensor_guid_t const &sgd_v) {
  TaskBinding b;
  b.bind(WEIGHT, TensorGuidSpec{weight, IsGrad::YES});
  b.bind(WEIGHT, TensorGuidSpec{weight, IsGrad::NO});
  if (attrs.momentum > 0.0f) {
    b.bind(SGD_V, TensorGuidSpec{sgd_v, IsGrad::YES});
  }
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    b.bind_arg(HANDLE, ff_handle());
    return {task_id_t::SGD_UPD_NCCL_TASK_ID, b};
  } else {
    return {task_id_t::SGD_UPD_PS_TASK_ID, b};
  }
}

static void sgd_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<SGDOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(WEIGHT);
  auto weight = acc.get_tensor<Permissions::RW>(WEIGHT);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  assert(weight.shape == weight_grad.shape);
  size_t size = weight_grad.shape.get_volume();

  assert(weight_grad.shape.get_volume() & weight.shape.get_volume() == 0);
  size_t num_replicas =
      weight_grad.shape.get_volume() / weight.shape.get_volume();

  float *sgd_v_ptr;
  if (attrs.momentum > 0.0f) {
    auto sgd_v = acc.get_tensor<Permissions::RW>(SGD_V);
    assert(sgd_v.shape == weight.shape);
    sgd_v_ptr = sgd_v.get_float_ptr();
  }

  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    auto handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
    profile(sgd_nccl_update_task_gpu,
            profiling,
            "[SGD NCCL] update_time = %.2lfms\n",
            attrs.lr,
            attrs.momentum,
            attrs.nesterov,
            attrs.weight_decay,
            handle,
            weight_grad.get_float_ptr(),
            size,
            weight.get_float_ptr(),
            sgd_v_ptr);

  } else {
    profile(sgd_ps_update_task_gpu,
            profiling,
            "[SGD PS] update_time = %.2lfms\n",
            attrs.lr,
            attrs.momentum,
            attrs.nesterov,
            attrs.weight_decay,
            weight_grad.get_float_ptr(),
            size,
            num_replicas,
            weight.get_float_ptr(),
            sgd_v_ptr);
  }
}

TaskImplFunction get_sgd_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{sgd_update_task_impl}};
}

TaskSignature get_adam_update_signature() {
  TaskSignature sig = make_empty_task_signature();
  add_slot(sig, WEIGHT, IsGrad::YES);
  add_slot(sig, WEIGHT, IsGrad::NO);
  add_slot(sig, ADAM_V, IsGrad::YES);
  add_slot(sig, ADAM_M, IsGrad::YES);
  add_arg_slot<AdamOptimizerAttrs>(sig, ATTRS);
  add_arg_slot<ProfilingSettings>(sig, PROFILING);
  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    add_unchecked_arg_slot<PerDeviceFFHandle>(sig, HANDLE);
  }
  return sig;
}

TaskInvocation adam_update(AdamOptimizerAttrs const &attrs,
                           tensor_guid_t const &weight,
                           tensor_guid_t const &adam_v,
                           tensor_guid_t const &adam_m) {
  TaskBinding b;
  b.bind(WEIGHT, TensorGuidSpec{weight, IsGrad::YES});
  b.bind(WEIGHT, TensorGuidSpec{weight, IsGrad::NO});
  b.bind(ADAM_M, TensorGuidSpec{adam_m, IsGrad::YES});
  b.bind(ADAM_V, TensorGuidSpec{adam_v, IsGrad::YES});
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    b.bind_arg(HANDLE, ff_handle());
    return {task_id_t::ADAM_UPD_NCCL_TASK_ID, b};
  }
  return {task_id_t::ADAM_UPD_PS_TASK_ID, b};
}

static void adam_update_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<AdamOptimizerAttrs>(ATTRS);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(WEIGHT);
  auto weight = acc.get_tensor<Permissions::RW>(WEIGHT);
  auto v_tensor = acc.get_tensor<Permissions::RW>(ADAM_V);
  auto m_tensor = acc.get_tensor<Permissions::RW>(ADAM_M);

  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  assert(weight.shape == weight_grad.shape);
  size_t size = weight_grad.shape.get_volume();

  assert(weight_grad.shape.get_volume() % weight.shape.get_volume() == 0);
  size_t num_replicas =
      weight_grad.shape.get_volume() / weight.shape.get_volume();

  if (CHOSEN_SYNC_TYPE == ParamSync::NCCL) {
    auto handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
    profile(adam_nccl_update_task_gpu,
            profiling,
            "[Adam NCCL] update_time = %.2lfms\n",
            attrs.alpha_t,
            attrs.beta1,
            attrs.beta2,
            attrs.weight_decay,
            attrs.epsilon,
            size,
            handle,
            weight_grad.get_float_ptr(),
            m_tensor.get_float_ptr(),
            v_tensor.get_float_ptr(),
            weight.get_float_ptr());
  } else {
    profile(adam_ps_update_task_gpu,
            profiling,
            "[Adam NCCL] update_time = %.2lfms\n",
            attrs.alpha_t,
            attrs.beta1,
            attrs.beta2,
            attrs.weight_decay,
            attrs.epsilon,
            size,
            num_replicas,
            weight_grad.get_float_ptr(),
            m_tensor.get_float_ptr(),
            v_tensor.get_float_ptr(),
            weight.get_float_ptr());
  }
}

TaskImplFunction get_adam_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{adam_update_task_impl}};
}

TaskSignature get_update_signature(OptimizerAttrs const &attrs) {
  return attrs.visit<TaskSignature>(overload{
      [&](SGDOptimizerAttrs const &) { return get_sgd_update_signature(); },
      [&](AdamOptimizerAttrs const &) { return get_adam_update_signature(); }});
}

TaskInvocation get_update_invocation(
    OptimizerAttrs const &attrs,
    tensor_guid_t const &weight,
    std::vector<tensor_guid_t> const &grad_buffer_tensors) {
  return attrs.visit<TaskInvocation>(overload{
      [&](SGDOptimizerAttrs const &s) {
        return sgd_update(s, weight, grad_buffer_tensors.at(0));
      },
      [&](AdamOptimizerAttrs const &s) {
        return adam_update(
            s, weight, grad_buffer_tensors.at(0), grad_buffer_tensors.at(1));
      }});
}

TaskImplFunction get_update_task_impl(OptimizerAttrs const &attrs) {
  return attrs.visit<TaskImplFunction>(overload{
      [&](SGDOptimizerAttrs const &) { return get_sgd_update_task_impl(); },
      [&](AdamOptimizerAttrs const &) { return get_adam_update_task_impl(); }});
}

} // namespace FlexFlow
