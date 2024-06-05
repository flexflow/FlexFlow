/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "kernels/dropout_kernels.h"
#include "device.h"
#include "kernels/ff_handle.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
namespace Kernels {
namespace Dropout {

DropoutPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                  float rate,
                                  unsigned long long seed,
                                  ArrayShape output_shape,
                                  Allocator allocator) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffDropoutDescriptor_t dropoutDesc;
  void *reserveSpace;
  void *dropoutStates;
  size_t reserveSpaceSize;
  size_t dropoutStateSize;
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(miopenDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, output_shape));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_shape));
  checkCUDNN(
      miopenDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    dropoutStates = allocator.allocate(totalSize);
    reserveSpace = ((char *)dropoutStates) + dropoutStateSize;
  }
  checkCUDNN(miopenSetDropoutDescriptor(dropoutDesc,
                                        handle.dnn,
                                        rate,
                                        dropoutStates,
                                        dropoutStateSize,
                                        seed,
                                        false,
                                        false,
                                        MIOPEN_RNG_PSEUDO_XORWOW));
  DropoutPerDeviceState per_device_state = {handle,
                                            allocator,
                                            inputTensor,
                                            outputTensor,
                                            dropoutDesc,
                                            reserveSpace,
                                            dropoutStates,
                                            reserveSpaceSize,
                                            dropoutStateSize};
  return per_device_state;
}

void forward_kernel(hipStream_t stream,
                    DropoutPerDeviceState &m,
                    float const *input_ptr,
                    float *output_ptr) {
  checkCUDNN(miopenSetStream(m.handle.dnn, stream));

  checkCUDNN(miopenDropoutForward(m.handle.dnn,
                                  m.dropoutDesc,
                                  m.inputTensor /* not used */,
                                  m.inputTensor,
                                  input_ptr,
                                  m.outputTensor,
                                  output_ptr,
                                  m.reserveSpace,
                                  m.reserveSpaceSize));
}

void backward_kernel(hipStream_t stream,
                     DropoutPerDeviceState &m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  checkCUDNN(miopenSetStream(m.handle.dnn, stream));

  checkCUDNN(miopenDropoutBackward(m.handle.dnn,
                                   m.dropoutDesc,
                                   m.inputTensor /* not used */,
                                   m.outputTensor,
                                   output_grad_ptr,
                                   m.inputTensor,
                                   input_grad_ptr,
                                   m.reserveSpace,
                                   m.reserveSpaceSize));
}

void cleanup_kernel(Allocator allocator,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffDropoutDescriptor_t dropoutDesc,
                    void *dropoutStates) {
  allocator.deallocate(dropoutStates);
  checkCUDNN(miopenDestroyTensorDescriptor(inputTensor));
  checkCUDNN(miopenDestroyTensorDescriptor(outputTensor));
  checkCUDNN(miopenDestroyDropoutDescriptor(dropoutDesc));
}

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow
