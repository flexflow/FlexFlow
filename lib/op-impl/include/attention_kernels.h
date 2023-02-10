#ifndef _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class MultiHeadAttentionMeta : public OpMeta {
public:
  MultiHeadAttentionMeta(FFHandler handler,
                         MultiHeadAttention const *attn,
                         Legion::Memory gpu_mem,
                         int num_samples,
                         int num_heads);
  ~MultiHeadAttentionMeta(void);

public:
  Realm::RegionInstance reserveInst;
  size_t weightSize, reserveSpaceSize;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnAttnDescriptor_t attnDesc;
  cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
#endif
  int *devQoSeqArray, *devKvSeqArray, *loWinIdx, *hiWinIdx;
  void *reserveSpace;
};

namespace Kernels {
namespace MultiHeadAttention {
   void forward_kernel_wrapper(MultiHeadAttentionMeta const *m,
                                     float const *query_ptr,
                                     float const *key_ptr,
                                     float const *value_ptr,
                                     float const *weight_ptr,
                                     float *output_ptr);
   void backward_kernel_wrapper(MultiHeadAttentionMeta const *m,
                                      float const *query_ptr,
                                      float *query_grad_ptr,
                                      float const *key_ptr,
                                      float *key_grad_ptr,
                                      float const *value_ptr,
                                      float *value_grad_ptr,
                                      float const *weight_ptr,
                                      float *weight_grad_ptr,
                                      float const *output_grad_ptr);

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

} // namespace Internal
} // namespace MultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H