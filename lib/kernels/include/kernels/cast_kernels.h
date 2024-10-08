#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Cast {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type);

} // namespace FlexFlow::Kernels::Cast

#endif
