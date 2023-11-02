#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Transpose;

class TransposeMeta : public OpMeta {
public:
  TransposeMeta(FFHandler handler, Transpose const *transpose);
  int num_dim;
  int perm[MAX_TENSOR_DIM];
};

namespace Kernels {
namespace Transpose {

void forward_kernel_wrapper(TransposeMeta const *m,
                            float const *input_ptr,
                            float *output_ptr,
                            Legion::Domain in_domain,
                            Legion::Domain out_domain);
void backward_kernel_wrapper(TransposeMeta const *m,
                             float *input_grad_ptr,
                             float const *output_grad_ptr,
                             Legion::Domain in_grad_domain,
                             Legion::Domain out_grad_domain);

namespace Internal {

void forward_kernel(TransposeMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    Legion::Domain in_domain,
                    Legion::Domain out_domain,
                    ffStream_t stream);
void backward_kernel(TransposeMeta const *m,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     Legion::Domain in_grad_domain,
                     Legion::Domain out_grad_domain,
                     ffStream_t stream);

} // namespace Internal
} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
