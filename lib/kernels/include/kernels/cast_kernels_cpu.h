#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Cast {
namespace CPU {

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type);

void backward_kernel(GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type);

} // namespace CPU
} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

#endif
