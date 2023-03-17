#ifndef _FLEXFLOW_KERNELS_CUDA_ATTENTION_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_ATTENTION_KERNELS_H

#include "kernels/attention_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace MultiHeadAttention {
namespace Internal {

void forward_kernel(MultiHeadAttentionMeta const *m,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    ffStream_t stream);
void backward_kernel(MultiHeadAttentionMeta const *m,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr,
                     ffStream_t stream);

} 
}
}
}

#endif
