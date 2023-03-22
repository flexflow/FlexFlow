#ifndef _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"

namespace FlexFlow {

class EmbeddingMeta : public OpMeta {
public:
  EmbeddingMeta(FFHandler handle, Op const *op);
  DataType input_data_type;
  AggrMode aggr;
};

namespace Kernels {
namespace Embedding {

/* struct GenericTensorPtr { */
/* public: */
/*   int32_t const *get_int32_ptr() const; */
/*   int64_t const *get_int64_ptr() const; */
/*   float const *get_float_ptr() const; */
/*   double const *get_double_ptr() const; */
/*   half const *get_half_ptr() const; */

/*   int32_t *get_int32_ptr(); */
/*   int64_t *get_int64_ptr(); */
/*   float *get_float_ptr(); */
/*   double *get_double_ptr(); */
/*   half *get_half_ptr(); */
/* private: */
/*   DataType data_type; */
/*   void *ptr; */
/* }; */

void forward_kernel_wrapper(EmbeddingMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight,
                            int in_dim,
                            int out_dim,
                            int batch_size);
void backward_kernel_wrapper(EmbeddingMeta const *m,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorW const &weight_grad,
                             int in_dim,
                             int out_dim,
                             int batch_size);

void rand_generate_int64_wrapper(int64_t *ptr, size_t size, int64_t p);
void rand_generate_int32_wrapper(int32_t *ptr, size_t size, int32_t p);

namespace Internal {
template <typename TI, typename TD>
void forward_kernel(TI const *input_ptr,
                    TD *output_ptr,
                    TD const *weight_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    AggrMode aggr,
                    int outputSize,
                    ffStream_t stream);

template <typename TI, typename TD>
void backward_kernel(TI const *input_ptr,
                     TD const *output_ptr,
                     TD *weight_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size,
                     AggrMode aggr,
                     int outputSize,
                     ffStream_t stream);
template <typename TD>
__global__ void rand_generate_int(TD *ptr, size_t size, TD p);
} // namespace Internal
} // namespace Embedding
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
