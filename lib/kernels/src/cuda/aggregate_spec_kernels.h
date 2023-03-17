#ifndef _FLEXFLOW_KERNELS_CUDA_AGGREGATE_SPEC_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_AGGREGATE_SPEC_KERNELS_H

namespace FlexFlow {
namespace Kernels {
namespace AggregateSpec {
namespace Internal {

__global__ void aggspec_forward_kernel(float **exp_preds,
                           int const *exp_assign,
                           float *output,
                           int n,           // num experts
                           int const k,     // num chosen experts
                           int exp_samples, // max samples per expert
                           int const batch_size,
                           int out_dim);
__global__ void aggspec_backward_kernel(float **exp_grads,
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

}
}
}
}

#endif
