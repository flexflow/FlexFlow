#ifndef _FLEXFLOW_KERNELS_CUDA_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_BATCH_MATMUL_KERNELS_H

#include "kernels/batch_matmul_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace BatchMatmul {
namespace Internal {

void forward_kernel(BatchMatmulMeta const *meta,
                    float *o_ptr,
                    float const *a_ptr,
                    float const *b_ptr,
                    float const *c_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    ffStream_t stream,
                    int a_seq_length_dim = -1,
                    int b_seq_length_dim = -1,
                    int seq_length = -1);
void backward_kernel(BatchMatmulMeta const *meta,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     float *c_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch,
                     ffStream_t stream);

}
}
}
}

#endif
