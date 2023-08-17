#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Gather;

class GatherMeta : public OpMeta {
public:
  GatherMeta(FFHandler handler, Gather const *gather);

public:
  int legion_dim;
};

namespace Kernels {
namespace Gather {
void forward_kernel_wrapper(GatherMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &index,
                            GenericTensorAccessorW const &output);
void backward_kernel_wrapper(GatherMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &index,
                             GenericTensorAccessorW const &input_grad);
namespace Internal {
template <typename IndexType>
void forward_kernel(float const *input_ptr,
                    IndexType const *index_ptr,
                    float *output_ptr,
                    Legion::coord_t output_size,
                    Legion::coord_t stride,
                    Legion::coord_t input_dim_size,
                    Legion::coord_t output_dim_size,
                    ffStream_t stream);
template <typename IndexType>
void backward_kernel(float const *output_grad_ptr,
                     IndexType const *index_ptr,
                     float *input_grad_ptr,
                     Legion::coord_t output_size,
                     Legion::coord_t stride,
                     Legion::coord_t input_dim_size,
                     Legion::coord_t output_dim_size,
                     ffStream_t stream);
} // namespace Internal
} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
