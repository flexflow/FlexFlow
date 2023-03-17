#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"
#include "legion.h"

namespace FlexFlow {

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle,
                Legion::Memory gpu_mem,
                int output_n,
                int output_c,
                int output_h,
                int output_w, 
                bool relu,
                bool profiling);
  ~BatchNormMeta(void);
  Realm::RegionInstance reserveInst;
  ffTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  ffActivationDescriptor_t actiDesc;
  ffBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
  bool profiling;
};

namespace Kernels {
namespace BatchNorm {

void forward_kernel_wrapper(BatchNormMeta *m,
                            float const *input_ptr,
                            float *output_ptr,
                            float const *scale_ptr,
                            float const *bias_ptr);

void backward_kernel_wrapper(BatchNormMeta *m,
                             float const *input_ptr,
                             float *output_grad_ptr,
                             float const *output_ptr,
                             float *input_grad_ptr,
                             float const *scale_ptr,
                             float *scale_grad_ptr,
                             float *bias_grad_ptr,
                             size_t numElements);

}
}
}

#endif
