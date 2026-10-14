/* Copyright 2023 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/device.h"
#include "kernels/reduce_kernels_gpu.h"

namespace FlexFlow {
namespace Kernels {
namespace Reduce {

ReducePerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                     ReduceOp const &op_type,
                                     size_t const &reduction_size,
                                     TensorShape const &input_shape,
                                     TensorShape const &output_shape) {

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffReduceTensorDescriptor_t reduceDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));

  checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceDesc));

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));

  ReducePerDeviceState per_device = ReducePerDeviceState{
      /*handle=*/handle,
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*reduceDesc=*/reduceDesc,
      /*op_type=*/op_type,
      /*reduction_size=*/reduction_size,
  };
  return per_device;
}

void gpu_forward_kernel(cudaStream_t stream,
                        ReducePerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnReduceTensor(m.handle.dnn,
                               m.reduceDesc,
                               nullptr /*indices*/,
                               0 /*indicesSizeInBytes*/,
                               m.handle.workSpace,
                               m.handle.workSpaceSize,
                               &alpha,
                               m.inputTensor,
                               input_ptr,
                               &beta,
                               m.outputTensor,
                               output_ptr));
};

void gpu_backward_kernel(cudaStream_t stream,
                         ReducePerDeviceState const &m,
                         float const *output_grad_ptr,
                         float *input_grad_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));
  float alpha = 1.0, beta = 1.0f;
  switch (m.reduce_op) {
    case ReduceOp::SUM:
      alpha = 1.0f;
      break;
    case ReduceOp::AVG:
      // When the output is the average of multiple input elements
      // we need to scale the gradients by 1.0 / reduction_size
      alpha = 1.0f / m.reduction_size;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnAddTensor(m.handle.dnn,
                            &alpha,
                            m.outputTensor,
                            output_grad_ptr,
                            &beta,
                            m.inputTensor,
                            input_grad_ptr));
}

} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow
