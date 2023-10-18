#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class BatchMatmul;

class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handler, BatchMatmul const *bmm);
  int a_seq_length_dim, b_seq_length_dim;
};

namespace Kernels {
namespace BatchMatmul {
void forward_kernel_wrapper(BatchMatmulMeta const *meta,
                            float *o_ptr,
                            float const *a_ptr,
                            float const *b_ptr,
                            float const *c_ptr,
                            int m,
                            int n,
                            int k,
                            int batch,
                            int a_seq_length_dim = -1,
                            int b_seq_length_dim = -1,
                            int seq_length = -1);
void backward_kernel_wrapper(BatchMatmulMeta const *meta,
                             float const *o_ptr,
                             float const *o_grad_ptr,
                             float const *a_ptr,
                             float *a_grad_ptr,
                             float const *b_ptr,
                             float *b_grad_ptr,
                             float *c_grad_ptr,
                             int m,
                             int n,
                             int k,
                             int batch);

namespace Internal {

void forward_kernel(BatchMatmulMeta const *meta,
                    float *o_ptr,
                    float const *a_ptr,
                    float const *b_ptr,
                    float const *c_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    ffStream_t stream,
                    int a_seq_length_dim = -1,
                    int b_seq_length_dim = -1,
                    int seq_length = -1);
void backward_kernel(BatchMatmulMeta const *meta,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     float *c_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch,
                     ffStream_t stream);
} // namespace Internal
} // namespace BatchMatmul
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
