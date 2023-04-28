#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H

#include "accessor.h"
#include "utils/variant.h"

namespace FlexFlow {

void uniform_init_kernel(GenericTensorAccessorW const &tensor,
                         size_t seed,
                         float min_val, 
                         float max_val);
void glorot_uniform_init_kernel(GenericTensorAccessorW const &tensor,
                                size_t seed,
                                float scale_factor);
void norm_init_kernel(GenericTensorAccessorW const &tensor,
                      size_t seed,
                      float mean,
                      float stddev);
void zero_init_kernel(GenericTensorAccessorW const &tensor);
void constant_init_kernel(GenericTensorAccessorW const &tensor,
                          DataTypeValue value);

}

#endif
