#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/parallel_ops/combine.h"

namespace FlexFlow {

class CombineMeta : public OpMeta {
public:
  CombineMeta(FFHandler handle);
  DataType data_type;
};

namespace Kernels {
namespace Combine {

template <typename T>
void forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements);

} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
