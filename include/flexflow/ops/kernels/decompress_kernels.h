#ifndef _FLEXFLOW_DECOMPRESS_KERNELS_H
#define _FLEXFLOW_DECOMPRESS_KERNELS_H

#include "flexflow/device.h"

namespace FlexFlow {
namespace Kernels {

namespace Decommpress {
template <typename T1, typename T2>
void decompress_weight_bias(T1 *input_weight_ptr,
                            T2 *weight_ptr,
                            T2 *params,
                            int group_size,
                            int tensor_size);
}

} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_DECOMPRESS_KERNELS_H
