#ifndef _FLEXFLOW_OPS_KERNELS_ALLREDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ALLREDUCE_KERNELS_H

#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/parallel_ops/allreduce.h"

namespace FlexFlow {

class AllReduceMeta : public OpMeta {
public:
  AllReduceMeta(FFHandler handle, AllReduce const *reduct);
};

namespace Kernels {
namespace AllReduce {

void forward_kernel_wrapper(AllReduceMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void backward_kernel_wrapper(AllReduceMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

void inference_kernel_wrapper(AllReduceMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);

void peft_bwd_kernel_wrapper(AllReduceMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);
} // namespace AllReduce
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_ALLREDUCE_KERNELS_H
