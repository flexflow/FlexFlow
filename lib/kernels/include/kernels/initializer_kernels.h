#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H

#include "accessor.h"
#include "kernels/cpu.h"
#include "utils/variant.h"
#include "op-attrs/datatype_value.dtg.h"

namespace FlexFlow {

void uniform_init_kernel(GenericTensorAccessorW const &,
                         size_t seed,
                         float min_val,
                         float max_val);
void glorot_uniform_init_kernel(GenericTensorAccessorW const &,
                                size_t seed,
                                float scale_factor);
void norm_init_kernel(GenericTensorAccessorW const &,
                      size_t seed,
                      float mean,
                      float stddev);
void zero_init_kernel(TaskLocation const &, GenericTensorAccessorW const &);
void zero_init_kernel_gpu(GenericTensorAccessorW const &);
void zero_init_kernel_cpu(GenericTensorAccessorW const &);

void constant_init_kernel(TaskLocation const &,
                          GenericTensorAccessorW const &,
                          DataTypeValue);
void constant_init_kernel_gpu(GenericTensorAccessorW const &, DataTypeValue);
void constant_init_kernel_cpu(GenericTensorAccessorW const &, DataTypeValue);

} // namespace FlexFlow

#endif
