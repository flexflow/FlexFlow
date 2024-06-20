#ifndef _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {
using Legion::coord_t;

class RMSNorm;

class RMSNormMeta : public OpMeta {
public:
  RMSNormMeta(FFHandler handler,
              RMSNorm const *rms,
              MemoryAllocator &gpu_mem_allocator);
  ~RMSNormMeta(void);
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

  int in_dim;
  int batch_size;
  int num_elements;
  Realm::RegionInstance reserveInst;
  // PEFT related fields
  void *input_activation;
  size_t allocated_peft_buffer_size = 0;
};

namespace Kernels {
namespace RMSNorm {
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output);
void inference_kernel_wrapper(RMSNormMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorR const &weight,
                              GenericTensorAccessorW const &output);
void backward_kernel_wrapper(RMSNormMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &weight,
                             GenericTensorAccessorW const &weight_grad);
void peft_bwd_kernel_wrapper(RMSNormMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &weight);
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
