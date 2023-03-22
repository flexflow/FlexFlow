#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"
#include "legion.h"

namespace FlexFlow {

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle){};
  int legion_axis;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace Concat {

void init_meta(ConcatMeta *meta, int legion_axis);

template <int DIM>
void forward_kernel_wrapper(ConcatMeta const *m,
                            float *output,
                            float const * const *inputs,
                            Legion::Rect<DIM> output_domain,
                            Legion::Rect<DIM> const *input_domains,
                            int num_inputs,
                            int axis);

template <int DIM>
void backward_kernel_wrapper(float const *output_grad,
                             float * const *input_grads,
                             Legion::Rect<DIM> output_domain,
                             Legion::Rect<DIM> const *input_domains,
                             int num_inputs,
                             int axis);

namespace Internal {

template <int DIM>
void forward_kernel(float *output,
                    float const * const *inputs,
                    Legion::Rect<DIM> output_domain,
                    Legion::Rect<DIM> const *input_domains,
                    int num_inputs,
                    int axis,
                    ffStream_t stream);

template <int DIM>
void backward_kernel(float const *output_grad,
                     float * const *input_grads,
                     Legion::Rect<DIM> output_domain,
                     Legion::Rect<DIM> const *input_domains,
                     int num_inputs,
                     int axis,
                     ffStream_t stream);
} // namespace Internal
} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
