#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

void sgd_ps_update_task_gpu(ffStream_t,
                            float lr,
                            float momentum,
                            bool nesterov,
                            float weight_decay,
                            float const *weight_grad_ptr,
                            size_t size,
                            int num_replicas,
                            float *weight_ptr,
                            float *sgd_v_ptr);

void sgd_nccl_update_task_gpu(ffStream_t,
                              float lr,
                              float momentum,
                              bool nesterov,
                              float weight_decay,
                              PerDeviceFFHandle const &,
                              float const *weight_grad_ptr,
                              size_t size,
                              float *weight_ptr,
                              float *sgd_v_ptr);

void adam_ps_update_task_gpu(ffStream_t,
                             float alpha_t,
                             float beta1,
                             float beta2,
                             float weight_decay,
                             float epsilon,
                             float const *weight_grad_ptr,
                             float *adam_m_ptr,
                             float *adam_v_ptr,
                             float *weight_ptr);

void adam_nccl_update_task_gpu(ffStream_t,
                               float alpha_t,
                               float beta1,
                               float beta2,
                               float weight_decay,
                               float epsilon,
                               PerDeviceFFHandle const &,
                               float const *weight_grad_ptr,
                               float *adam_m_ptr,
                               float *adam_v_ptr,
                               float *weight_ptr);

} // namespace FlexFlow

#endif
