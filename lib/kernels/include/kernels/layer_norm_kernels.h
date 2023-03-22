#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "device.h"
#include "fftype.h"
#include "op_meta.h"

namespace FlexFlow {


class LayerNormMeta : public OpMeta {
public:
  LayerNormMeta(FFHandler handle, LayerNorm const *ln);

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  float *mean_ptr, *rstd_ptr, *ds_ptr, *db_ptr, *scale_ptr, *bias_ptr;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace LayerNorm {
template <typename T>
void forward_kernel_wrapper(LayerNormMeta const *m,
                                     T const *input_ptr,
                                     T *output_ptr,
                                     T *gamma_ptr,
                                     T *beta_ptr);

template <typename T>
void backward_kernel_wrapper(LayerNormMeta const *m,
                                      T const *output_grad_ptr,
                                      T const *input_ptr,
                                      T *input_grad_ptr,
                                      T const *gamma_ptr,
                                      T *gamma_grad_ptr,
                                      T *beta_grad_ptr);

namespace Internal {

template <typename T>
void forward_kernel(LayerNormMeta const *m,
                            T const *input_ptr,
                            T *output_ptr,
                            T *gamma_ptr,
                            T *beta_ptr,
                            ffStream_t stream);

template <typename T>
void backward_kernel(LayerNormMeta const *m,
                            T const *output_grad_ptr,
                            T const *input_ptr,
                            T *input_grad_ptr,
                            T const *gamma_ptr,
                            T *gamma_grad_ptr,
                            T *beta_grad_ptr,
                            ffStream_t stream);
} // namespace Internal
} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
