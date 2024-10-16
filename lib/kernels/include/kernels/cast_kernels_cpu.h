#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Cast {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        DataType input_type,
                        DataType output_type);

void cpu_backward_kernel(GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &output,
                         DataType input_type,
                         DataType output_type);

} // namespace FlexFlow::Kernels::Cast

#endif
