#ifndef _FLEXFLOW_KERNELS_CUDA_AGGREGATE_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_AGGREGATE_KERNELS_H

namespace FlexFlow {
namespace Kernels {
namespace Aggregate {
namespace Internal {

__global__ void agg_forward_kernel(float **exp_preds,
                                   int const *exp_assign,
                                   float const *gate_net_preds,
                                   float *output,
                                   int n,
                                   int const k,     // num chosen experts
                                   int exp_samples, // max samples per expert
                                   int const batch_size,
                                   int out_dim);

__global__ void agg_backward_kernel(float **exp_preds,
                                    float **exp_grads,
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
