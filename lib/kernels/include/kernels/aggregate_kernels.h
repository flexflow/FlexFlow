#ifndef _FLEXFLOW_OPS_KERNELS_AGGREGATE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_AGGREGATE_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

namespace Kernels {
namespace Aggregate {

void forward_kernel(ffStream_t stream,
                    float **exp_preds,
                    int const *acc_gate_assign_ptr,
                    float const *acc_gate_pred_ptr,
                    float *acc_output_ptr,
                    int n,
                    int const k,
                    int rows,
                    int const batch_size,
                    int out_dim);

void backward_kernel(ffStream_t stream,
                     float **exp_preds,
                     float **exp_grads,
                     int const *acc_gate_assign_ptr,
                     int const *acc_true_gate_assign_ptr,
                     float const *acc_gate_pred_ptr,
                     float *full_acc_gate_grad_ptr,
                     float const *acc_output_grad_ptr,
                     int n,
                     int const k,
                     int rows,
                     float lambda_bal,
                     int const batch_size,
                     int out_dim);

} // namespace Aggregate
} // namespace Kernels
} // namespace FlexFlow

#endif
