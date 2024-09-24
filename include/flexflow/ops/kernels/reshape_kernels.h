#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Reshape;

class ReshapeMeta : public OpMeta {
public:
  ReshapeMeta(FFHandler handler, Reshape const *reshape);
  DataType data_type;
};

namespace Kernels {
namespace Reshape {

template <typename T>
void forward_kernel_wrapper(T const *input_ptr,
                            T *output_ptr,
                            size_t num_elements);

template <typename T>
void backward_kernel_wrapper(T *input_grad_ptr,
                             T const *output_grad_ptr,
                             size_t num_elements);

namespace Internal {

template <typename T>
void forward_kernel(T const *input_ptr,
                    T *output_ptr,
                    size_t num_elements,
                    ffStream_t stream);
template <typename T>
void backward_kernel(T *input_grad_ptr,
                     T const *output_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);

} // namespace Internal
} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
