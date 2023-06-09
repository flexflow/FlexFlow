#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class Pool2DPerDeviceState : public PerDeviceOpState {
public:
  Pool2DPerDeviceState(FFHandler handle);
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;
  ffPoolingDescriptor_t poolDesc;
  bool relu;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace Pool2D {

void init_kernel(Pool2DPerDeviceState *m,
                 int input_w,
                 int input_h,
                 int input_c,
                 int input_n,
                 int output_w,
                 int output_h,
                 int output_c,
                 int output_n,
                 int pad_h,
                 int pad_w,
                 int kernel_h,
                 int kernel_w,
                 int stride_h,
                 int stride_w,
                 PoolType pool_type);

void forward_kernel(ffStream_t stream,
                    Pool2DPerDeviceState const *m,
                    void const *input_ptr,
                    void *output_ptr);

void backward_kernel(ffStream_t stream,
                     Pool2DPerDeviceState const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr);

} // namespace Pool2D
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
