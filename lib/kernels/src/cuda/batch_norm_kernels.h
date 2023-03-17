#ifndef _FLEXFLOW_KERNELS_CUDA_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_BATCH_NORM_KERNELS_H

#include "kernels/batch_norm_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace BatchNorm {
namespace Internal {

void forward_kernel(BatchNormMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void backward_kernel(BatchNormMeta const *m,
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
}

#endif 
