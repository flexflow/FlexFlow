#ifndef _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"
#include "kernels/config.h"
#include "kernels/allocation.h"
#include <memory>
#include "op-attrs/ops/attention.h"

namespace FlexFlow {

class MHAPerDeviceState : public PerDeviceOpState {
public:
  MHAPerDeviceState(FFHandler handler,
                    std::unique_ptr<IAllocator> allocator,
                    int num_samples,
                    int num_heads,
                    int qSize,
                    int kSize,
                    int vSize,
                    int qProjSize,
                    int kProjSize,
                    int vProjSize,
                    int oProjSize,
                    int qoSeqLength,
                    int kvSeqLength,
                    bool add_bias_kv);
  MHAPerDeviceState(FFHandler handler, 
                    std::unique_ptr<IAllocator> allocator,
                    MultiHeadAttentionAttrs const &attrs);

public:
  size_t weightSize, reserveSpaceSize;
  ffAttnDescriptor_t attnDesc;
  ffSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
  int *devQoSeqArray, *devKvSeqArray, *loWinIdx, *hiWinIdx;
  void *reserveSpace;
  std::unique_ptr<IAllocator> allocator;
};


namespace Kernels {
namespace MultiHeadAttention {

void init_kernel(MHAPerDeviceState *m,
                 int num_samples,
                 int num_heads,
                 int qSize,
                 int kSize,
                 int vSize,
                 int qProjSize,
                 int kProjSize,
                 int vProjSize,
                 int oProjSize,
                 int qoSeqLength,
                 int kvSeqLength,
                 bool add_bias_kv);

void forward_kernel(ffStream_t stream,
                    MHAPerDeviceState const &device_state,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     MHAPerDeviceState const &device_state,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr);

} 
}
}

#endif 
