#ifndef _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_AGGREGATE_SPEC_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"

namespace FlexFlow {

#define AGGREGATE_SPEC_MAX_K 4
#define AGGREGATE_SPEC_MAX_BATCH_SIZE 32
#define AGGREGATE_SPEC_MAX_N 12

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
}
}
}

#endif 
