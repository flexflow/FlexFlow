#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_GROUPBY_H

#include "kernels/device.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GroupByPerDeviceState {
  float **dev_region_ptrs;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(GroupByPerDeviceState, dev_region_ptrs);

namespace Kernels {
namespace GroupBy {

GroupByPerDeviceState init_kernel( int n );

void forward_kernel(ffStream_t stream,
                    GroupByPerDeviceState &m,
                    float const *input,
                    int const *exp_assign,
                    float *outputs,
                    int n,       // num experts
                    int k,       // chosen experts
                    float alpha, // factor additional memory assigned
                    int batch_size,
                    int data_dim);

void backward_kernel(ffStream_t stream,
                     GroupByPerDeviceState &m,
                     float *input_grad,
                     int const *exp_assign,
                     float *output_grads,
                     int n,       // num experts
                     int k,       // chosen experts
                     float alpha, // factor additional memory assigned
                     int batch_size,
                     int data_dim);

void cleanup_kernel(float **dev_region_ptrs);

} // GroupBy
} // Kernels
} // namespace FlexFlow

#endif
