#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/softmax.h"

namespace FlexFlow {

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              Softmax const *softmax,
              Legion::Domain const &input_domain);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor;
  cudnnTensorDescriptor_t outputTensor;
#else
  miopenTensorDescriptor_t inputTensor;
  miopenTensorDescriptor_t outputTensor;
#endif
  bool profiling;
  bool inference_debugging;
  int dim;
};

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void backward_kernel_wrapper(SoftmaxMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

void inference_kernel_wrapper(SoftmaxMeta const *m,
                              BatchConfig const *bc,
                              bool is_last_op,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output,
                              GenericTensorAccessorW const &output_grad);

void peft_bwd_kernel_wrapper(SoftmaxMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad);

namespace Internal {
template <typename DT>
void forward_kernel(SoftmaxMeta const *m,
                    DT const *input_ptr,
                    DT *output_ptr,
                    ffStream_t stream);

template <typename DT>
void backward_kernel(SoftmaxMeta const *m,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);

template <typename DT>
void inference_kernel(SoftmaxMeta const *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int num_classes,
                      ffStream_t stream);

template <typename DT>
void peft_bwd_kernel(SoftmaxMeta const *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     int num_classes,
                     ffStream_t stream);

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
