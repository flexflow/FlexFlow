#include "kernels/optimizer_kernels.h"
#include "kernels/optimizer_kernels_cpu.h"
#include "kernels/optimizer_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

void sgd_update_task(device_stream_t const &stream,
                     device_handle_t const &handle,
                     float lr,
                     float momentum,
                     bool nesterov,
                     float weight_decay,
                     GenericTensorAccessorR const &weight_grad,
                     int num_replicas,
                     GenericTensorAccessorW const &weight,
                     std::optional<GenericTensorAccessorW> const &sgd_v) {
  ASSERT(sgd_v.has_value() == (momentum > 0.0f));

  if (stream.is_gpu()) {
    float *sgd_v_ptr = nullptr;
    if (momentum > 0.0f) {
      sgd_v_ptr = sgd_v.value().get_float_ptr();
    }

    PerDeviceFFHandle const &gpu_handle = handle.require_for_gpu();
#ifdef FF_USE_NCCL
    bool have_nccl_comm = (gpu_handle.ncclComm != nullptr);
#else
    bool have_nccl_comm = false;
#endif

    if (have_nccl_comm) {
      gpu_sgd_nccl_update_task(
          /*stream=*/stream.require_gpu(),
          /*lr=*/lr,
          /*momentum=*/momentum,
          /*nesterov=*/nesterov,
          /*weight_decay=*/weight_decay,
          /*handle=*/gpu_handle,
          /*weight_grad_ptr=*/weight_grad.get_float_ptr(),
          /*size=*/
          get_num_elements(weight_grad.shape.dims).int_from_positive_int(),
          /*weight_ptr=*/weight.get_float_ptr(),
          /*sgd_v_ptr=*/sgd_v_ptr);
    } else {
      // No communicator (single-rank), so the gradient allreduce is a no-op.
      gpu_sgd_ps_update_task(
          /*stream=*/stream.require_gpu(),
          /*lr=*/lr,
          /*momentum=*/momentum,
          /*nesterov=*/nesterov,
          /*weight_decay=*/weight_decay,
          /*weight_grad_ptr=*/weight_grad.get_float_ptr(),
          /*size=*/
          get_num_elements(weight_grad.shape.dims).int_from_positive_int(),
          /*num_replicas=*/num_replicas,
          /*weight_ptr=*/weight.get_float_ptr(),
          /*sgd_v_ptr=*/sgd_v_ptr);
    }
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_sgd_update_task(
        /*lr=*/lr,
        /*momentum=*/momentum,
        /*nesterov=*/nesterov,
        /*weight_decay=*/weight_decay,
        /*weight_grad=*/weight_grad,
        /*weight=*/weight,
        /*sgd_v=*/sgd_v);
  }
}

void adam_update_task(device_stream_t const &stream,
                      device_handle_t const &handle,
                      float alpha_t,
                      float beta1,
                      float beta2,
                      float weight_decay,
                      float epsilon,
                      float const *weight_grad_ptr,
                      size_t size,
                      int num_replicas,
                      float *weight_ptr,
                      float *adam_v_ptr,
                      float *adam_m_ptr) {
  if (stream.is_gpu()) {
    ASSERT(stream.is_cpu());
    gpu_adam_nccl_update_task(
        /*stream=*/stream.require_gpu(),
        /*alpha_t=*/alpha_t,
        /*beta1=*/beta1,
        /*beta2=*/beta2,
        /*weight_decay=*/weight_decay,
        /*epsilon=*/epsilon,
        /*handle=*/handle.require_for_gpu(),
        /*weight_grad_ptr=*/weight_grad_ptr,
        /*size=*/size,
        /*weight_ptr=*/weight_ptr,
        /*adam_v_ptr=*/adam_v_ptr,
        /*adam_m_ptr=*/adam_m_ptr);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_adam_update_task(
        /*alpha_t=*/alpha_t,
        /*beta1=*/beta1,
        /*beta2=*/beta2,
        /*weight_decay=*/weight_decay,
        /*epsilon=*/epsilon,
        /*weight_grad_ptr=*/weight_grad_ptr,
        /*size=*/size,
        /*num_replicas=*/num_replicas,
        /*weight_ptr=*/weight_ptr,
        /*adam_v_ptr=*/adam_v_ptr,
        /*adam_m_ptr=*/adam_m_ptr);
  }
}

} // namespace FlexFlow
