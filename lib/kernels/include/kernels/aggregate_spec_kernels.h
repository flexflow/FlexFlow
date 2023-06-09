#ifndef _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

#define AGGREGATE_SPEC_MAX_K 4
#define AGGREGATE_SPEC_MAX_BATCH_SIZE 32
#define AGGREGATE_SPEC_MAX_N 12

class AggregateSpecPerDeviceState : public PerDeviceOpState {
public:
  AggregateSpecPerDeviceState(PerDeviceFFHandle handle, int n);
  ~AggregateSpecPerDeviceState();
  float **dev_region_ptrs;
};

namespace Kernels {
namespace AggregateSpec {

void forward_kernel(ffStream_t stream,
                    AggregateSpecPerDeviceState const *m,
                    float **exp_preds,
                    int const *acc_gate_assign_ptr,
                    float *acc_output_ptr,
                    int n,
                    int const k,
                    int rows,
                    int const batch_size,
                    int out_dim);

void backward_kernel(ffStream_t stream,
                     AggregateSpecPerDeviceState const *m,
                     float **exp_grads,
                     int const *acc_gate_assign_ptr,
                     int const *acc_true_gate_assign_ptr,
                     float const *acc_gate_pred_ptr,
                     float *acc_full_gate_grad_ptr,
                     float const *acc_output_grad_ptr,
                     int n,
                     int const k,
                     int rows,
                     float lambda_bal,
                     int const batch_size,
                     int out_dim);
} // namespace AggregateSpec
} // namespace Kernels
} // namespace FlexFlow

#endif
