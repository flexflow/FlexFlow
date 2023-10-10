#ifndef _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H

#include "flexflow/accessor.h"
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

  float alpha;
  float beta;

  int in_dim;
  int batch_size;
  int num_elements;
  Realm::RegionInstance reserveInst;
};

namespace Kernels {
namespace RMSNorm {
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output);
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
