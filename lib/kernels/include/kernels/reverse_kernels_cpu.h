#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Reverse {

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor);

void cpu_backward_kernel(GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor);
} // namespace FlexFlow::Kernels::Reverse

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H
