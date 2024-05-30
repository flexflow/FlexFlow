#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle){};
  int legion_axis;
};

namespace Kernels {
namespace Concat {

void init_meta(ConcatMeta *meta, int legion_axis);
void forward_kernel_wrapper(ConcatMeta const *m,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const *inputs,
                            int num_inputs,
                            int axis);
void backward_kernel_wrapper(ConcatMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const *input_grads,
                             int num_inputs,
                             int axis);

namespace Internal {

void forward_kernel(GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const *inputs,
                    int num_inputs,
                    int axis,
                    ffStream_t stream);

void backward_kernel(GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const *input_grads,
                     int num_inputs,
                     int axis,
                     ffStream_t stream);
} // namespace Internal
} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
