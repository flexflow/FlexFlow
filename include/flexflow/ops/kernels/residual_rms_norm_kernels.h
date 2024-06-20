#ifndef _FLEXFLOW_OPS_KERNELS_RESIDUAL_RMSNORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESIDUAL_RMSNORM_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {
using Legion::coord_t;

class ResidualRMSNorm;

class ResidualRMSNormMeta : public OpMeta {
public:
  ResidualRMSNormMeta(FFHandler handler,
                      ResidualRMSNorm const *rms,
                      MemoryAllocator &gpu_mem_allocator);
  ~ResidualRMSNormMeta(void);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnReduceTensorDescriptor_t reduceDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenReduceTensorDescriptor_t reduceDesc;
#endif

public:
  float eps;
  void *rms_ptr;
  void *norm_ptr;

  bool inplace_residual;
  int in_dim;
  int batch_size;
  int num_elements;
  Realm::RegionInstance reserveInst;
  // PEFT related fields
  void *input_activation;
  size_t allocated_peft_buffer_size = 0;
};

namespace Kernels {
namespace ResidualRMSNorm {
void forward_kernel_wrapper(ResidualRMSNormMeta const *m,
                            GenericTensorAccessorR const &input1,
                            GenericTensorAccessorR const &input2,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &residual_output,
                            GenericTensorAccessorW const &output);
void inference_kernel_wrapper(ResidualRMSNormMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input1,
                              GenericTensorAccessorR const &input2,
                              GenericTensorAccessorR const &weight,
                              GenericTensorAccessorW const &residual_output,
                              GenericTensorAccessorW const &output);
void backward_kernel_wrapper(
    ResidualRMSNormMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &residual_output_rms_input,
    GenericTensorAccessorW const &residual_input0_grad,
    GenericTensorAccessorW const &residual_input1_grad,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &weight_grad);
void peft_bwd_kernel_wrapper(ResidualRMSNormMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorR const &output_grad_0,
                             GenericTensorAccessorR const &output_grad_1,
                             GenericTensorAccessorW const &input_grad_0,
                             GenericTensorAccessorW const &input_grad_1,
                             GenericTensorAccessorR const &weight);
} // namespace ResidualRMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
