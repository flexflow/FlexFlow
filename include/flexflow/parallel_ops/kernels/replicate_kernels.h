#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/parallel_ops/replicate.h"

namespace FlexFlow {

class ReplicateMeta : public OpMeta {
public:
  ReplicateMeta(FFHandler handle, Replicate const *repl);
};

namespace Kernels {
namespace Replicate {

template <typename T>
void forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements,
                     size_t num_replicas);

} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
