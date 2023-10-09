#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;
  ffPoolingDescriptor_t poolDesc;
  bool relu;
};

namespace Kernels {
namespace Pool2D {

void init_kernel(Pool2DMeta *m,
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

void forward_kernel_wrapper(Pool2DMeta const *m,
                            void const *input_ptr,
                            void *output_ptr);
void backward_kernel_wrapper(Pool2DMeta const *m,
                             void const *input_ptr,
                             void *input_grad_ptr,
                             void const *output_ptr,
                             void const *output_grad_ptr);

namespace Internal {

void forward_kernel(Pool2DMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    ffStream_t stream);

void backward_kernel(Pool2DMeta const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr,
                     ffStream_t stream);

} // namespace Internal
} // namespace Pool2D
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
