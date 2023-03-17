#ifndef _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H

#include "runtime/device.h"
#include "runtime/fftype.h"
#include "runtime/op_meta.h"

namespace FlexFlow {

class AggregateSpecMeta : public OpMeta {
public:
  AggregateSpecMeta(FFHandler handle, int n);
  ~AggregateSpecMeta(void);
  float **dev_region_ptrs;
};

namespace Kernels {
namespace AggregateSpec {
void forward_kernel_wrapper(AggregateSpecMeta const *m,
                                     float **exp_preds,
                                     int const *acc_gate_assign_ptr,
                                     float *acc_output_ptr,
                                     int n,
                                     int const k,
                                     int rows,
                                     int const batch_size,
                                     int out_dim);
void backward_kernel_wrapper(AggregateSpecMeta const *m,
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

namespace Internal {
void aggspec_forward_kernel(float **exp_preds,
                           int const *exp_assign,
                           float *output,
                           int n,           // num experts
                           int const k,     // num chosen experts
                           int exp_samples, // max samples per expert
                           int const batch_size,
                           int out_dim);
void aggspec_backward_kernel(float **exp_grads,
                            int const *exp_assign,
                            int const *true_exp_assign,
                            float const *gating_net_preds,
                            float *full_gating_grads,
                            float const *output_grads,
                            int n,           // num experts
                            int k,           // num chosen experts
                            int exp_samples, // max samples per expert
                            float lambda_bal,
                            int batch_size,
                            int out_dim);
} // namespace Internal
} // namespace AggregateSpec
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H
