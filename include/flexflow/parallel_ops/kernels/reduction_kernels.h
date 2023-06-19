#ifndef _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/parallel_ops/reduction.h"

namespace FlexFlow {

class ReductionMeta : public OpMeta {
public:
  ReductionMeta(FFHandler handle, Reduction const *reduct);
};

namespace Kernels {
namespace Reduction {

template <typename T>
void forward_kernel(T const *input_ptr,
                    T *output_ptr,
                    size_t num_elements,
                    size_t num_replicas);

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements);

} // namespace Reduction
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
