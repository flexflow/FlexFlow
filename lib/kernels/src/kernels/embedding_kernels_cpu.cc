#include "kernels/embedding_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Embedding {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &weight,
                        DataType input_data_type,
                        DataType output_data_type,
                        std::optional<AggregateOp> aggr,
                        num_tensor_dims_t in_dim,
                        num_tensor_dims_t out_dim,
                        int batch_size) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &weight_grad,
                         DataType output_data_type,
                         DataType input_data_type,
                         std::optional<AggregateOp> aggr,
                         num_tensor_dims_t in_dim,
                         num_tensor_dims_t out_dim,
                         int batch_size) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Embedding
