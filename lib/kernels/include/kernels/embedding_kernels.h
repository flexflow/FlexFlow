#ifndef _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow::Kernels::Embedding {
void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &weight,
                    DataType input_data_type,
                    DataType output_data_type,
                    std::optional<AggregateOp> aggr,
                    int in_dim,
                    int out_dim,
                    int batch_size);
void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &weight_grad,
                     DataType input_data_type,
                     DataType output_data_type,
                     std::optional<AggregateOp> aggr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

void rand_generate_int64_wrapper(int64_t *ptr, size_t size, int64_t p);
void rand_generate_int32_wrapper(int32_t *ptr, size_t size, int32_t p);

template <typename TD>
__global__ void rand_generate_int(TD *ptr, size_t size, TD p);

} // namespace FlexFlow::Kernels::Embedding

#endif // _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
