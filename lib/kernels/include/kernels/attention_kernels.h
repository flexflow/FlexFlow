#ifndef _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H

#include "device.h"
#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/ops/attention.h"
#include <memory>

namespace FlexFlow {

struct MHAPerDeviceState {
  PerDeviceFFHandle handle;
  size_t weightSize;
  size_t reserveSpaceSize;
  ffAttnDescriptor_t attnDesc;
  ffSeqDataDescriptor_t qDesc;
  ffSeqDataDescriptor_t kDesc;
  ffSeqDataDescriptor_t vDesc;
  ffSeqDataDescriptor_t oDesc;
  int *devQoSeqArray;
  int *devKvSeqArray;
  int *loWinIdx;
  int *hiWinIdx;
  void *reserveSpace;
  Allocator allocator;
};

FF_VISITABLE_STRUCT_NO_EQ(MHAPerDeviceState,
                          handle,
                          weightSize,
                          reserveSpaceSize,
                          attnDesc,
                          qDesc,
                          kDesc,
                          vDesc,
                          oDesc,
                          devQoSeqArray,
                          devKvSeqArray,
                          loWinIdx,
                          hiWinIdx,
                          reserveSpace,
                          allocator);

namespace Kernels {
namespace MultiHeadAttention {

MHAPerDeviceState init_kernel(PerDeviceFFHandle const &,
                              Allocator &,
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

void cleanup_kernel(Allocator &allocator,
                    MHAPerDeviceState const &device_state);

} // namespace MultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif
