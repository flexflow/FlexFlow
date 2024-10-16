#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/pool_2d.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Pool2DPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;
  ffPoolingDescriptor_t poolDesc;
  bool relu;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Pool2DPerDeviceState,
                                             handle,
                                             inputTensor,
                                             outputTensor,
                                             actiDesc,
                                             poolDesc,
                                             relu);

namespace Kernels::Pool2D {

Pool2DPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
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
                                 PoolOp pool_type);

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
                 PoolOp pool_type);

void forward_kernel(ffStream_t stream,
                    Pool2DPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr);

void backward_kernel(ffStream_t stream,
                     Pool2DPerDeviceState const &m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr);

} // namespace Kernels::Pool2D
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
