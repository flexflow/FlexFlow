#ifndef _FLEXFLOW_OPS_KERNELS_PARALLEL_IDENTITY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARALLEL_IDENTITY_KERNELS_H

#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/parallel_ops/parallel_identity.h"

namespace FlexFlow {

class ParallelIdentityMeta : public OpMeta {
public:
  ParallelIdentityMeta(FFHandler handle, ParallelIdentity const *reduct);
};

namespace Kernels {
namespace ParallelIdentity {

void forward_kernel_wrapper(ParallelIdentityMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void backward_kernel_wrapper(ParallelIdentityMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

void inference_kernel_wrapper(ParallelIdentityMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);

void peft_bwd_kernel_wrapper(ParallelIdentityMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);
} // namespace ParallelIdentity
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARALLEL_IDENTITY_KERNELS_H
