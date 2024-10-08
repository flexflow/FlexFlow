#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "device.h"
#include "ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/linear.h"

namespace FlexFlow {

struct LinearPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;
  float const *one_ptr; // how to handle this?
  cudnnActivationMode_t activation_mode;
  std::optional<Activation> activation;
  std::optional<RegularizerAttrs> regularizer;
  bool use_bias;
  DataType input_type, weight_type, output_type;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(LinearPerDeviceState,
                                             handle,
                                             outputTensor,
                                             actiDesc,
                                             one_ptr,
                                             activation_mode,
                                             activation,
                                             regularizer,
                                             use_bias,
                                             input_type,
                                             weight_type,
                                             output_type);

namespace Kernels::Linear {

LinearPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 float *one_ptr,
                                 std::optional<Activation> activation,
                                 std::optional<RegularizerAttrs> regularizer,
                                 bool use_bias,
                                 DataType input_type,
                                 DataType weight_type,
                                 DataType output_type,
                                 int batch_size,
                                 int channel);

bool use_activation(Activation activation);

void forward_kernel(ffStream_t stream,
                    LinearPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *filter_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);

void backward_kernel(ffStream_t stream,
                     LinearPerDeviceState const &m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

} // namespace Kernels::Linear
} // namespace FlexFlow

#endif
