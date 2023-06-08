#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class LinearPerDeviceState : public PerDeviceOpState {
public:
  LinearPerDeviceState(FFHandler handle, int batch_size);
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;

public:
  float const *one_ptr;
  ActiMode activation;
  optional<Regularizer> regularizer;
  bool use_bias;
  DataType input_type, weight_type, output_type;
};

namespace Kernels {
namespace Linear {
void init_kernel(LinearPerDeviceState *m, int batch_size, int channel);
bool use_activation(ActiMode mode);

void forward_kernel(ffStream_t stream,
                    LinearPerDeviceState const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *filter_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);
void backward_kernel(ffStream_t stream,
                     LinearPerDeviceState const *m,
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
                     
}
}
}

#endif
