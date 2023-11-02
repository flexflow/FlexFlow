#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Repartition;

class RepartitionMeta : public OpMeta {
public:
  RepartitionMeta(FFHandler handle, Repartition const *repart);
  DataType data_type;
};

namespace Kernels {
namespace Repartition {

template <typename T>
void forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements);

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
