#ifndef _FLEXFLOW_DECOMPRESS_KERNELS_H
#define _FLEXFLOW_DECOMPRESS_KERNELS_H

#include "flexflow/device.h"

namespace FlexFlow {
namespace Kernels {

template <typename DT>
__global__ void decompress_int4_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize);
template <typename DT>
__global__ void decompress_int8_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize);

template <typename DT>
__global__ void decompress_int4_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads);

template <typename DT>
__global__ void decompress_int8_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads);
// template <typename T1, typename T2>
// void decompress_weight_bias(T1 *input_weight_ptr,
//                             T2 *weight_ptr,
//                             T2 *params,
//                             int group_size,
//                             int tensor_size);

} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_DECOMPRESS_KERNELS_H
