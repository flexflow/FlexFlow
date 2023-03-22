#ifndef _FLEXFLOW_KERNELS_CUDA_ATTENTION_KERNELS_H
#define _FLEXFLOW_KERNELS_CUDA_ATTENTION_KERNELS_H

#include "kernels/attention_kernels.h"

namespace FlexFlow {

namespace Kernels {
namespace MultiHeadAttention {
namespace Internal {

void forward_kernel(FFHandler handler,
                    ffAttnDescriptor_t const &attnDesc,
                    int *loWinIdx,
                    int *hiWinIdx,
                    int *devQoSeqArray,
                    int *devKvSeqArray,
                    ffSeqDataDescriptor_t const &qDesc,
                    ffSeqDataDescriptor_t const &kDesc,
                    ffSeqDataDescriptor_t const &vDesc,
                    ffSeqDataDescriptor_t const &oDesc,
                    size_t weightSize,
                    void *reserveSpace, 
                    size_t reserveSpaceSize,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    cudaStream_t stream);
void backward_kernel(FFHandler const &handle,
                     ffAttnDescriptor_t const &attnDesc,
                     int *loWinIdx,
                     int *hiWinIdx,
                     int *devQoSeqArray,
                     int *devKvSeqArray,
                     ffSeqDataDescriptor_t const &qDesc,
                     ffSeqDataDescriptor_t const &kDesc,
                     ffSeqDataDescriptor_t const &vDesc,
                     ffSeqDataDescriptor_t const &oDesc,
                     size_t weightSize,
                     void *reserveSpace,
                     size_t reserveSpaceSize,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr,
                     cudaStream_t stream);

} 
}
}
}

#endif
