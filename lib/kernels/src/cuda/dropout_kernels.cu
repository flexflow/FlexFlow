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

// #include "kernels/cuda_helper.h"
#include "kernels/dropout_kernels.h"
#include "kernels/ff_handle.h"
#include "device.h"


namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;
using Legion::Memory;

namespace Kernels {
namespace Dropout {

DropoutPerDeviceState init_kernel(PerDeviceFFHandle handler,
                                  float rate,
                                  unsigned long long seed,
                                  ArrayShape output_domain,
                                  Allocator allocator) {
  Realm::RegionInstance reserveInst,
  ffTensorDescriptor_t inputTensor,
  ffTensorDescriptor_t outputTensor,
  ffDropoutDescriptor_t dropoutDesc,

  void *reserveSpace,
  void *dropoutState,
  size_t reserveSpaceSize,
  size_t dropoutStateSize
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(cudnnDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, output_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_domain));
  checkCUDNN(cudnnDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    dropoutStates = this->allocator->allocate(totalSize);
    reserveSpace = ((char *)dropoutStates) + dropoutStateSize;
  }
  checkCUDNN(cudnnSetDropoutDescriptor(
      dropoutDesc, handle.dnn, rate, dropoutStates, dropoutStateSize, seed));
  DropoutPerDeviceSate per_device_state = {
                                            handler,
                                            rate,
                                            seed,
                                            &output_domain,
                                            allocator,
                                            reserveInst,
                                            inputTensor,
                                            outputTensor,
                                            dropoutDesc,
                                            reserveSpace,
                                            dropoutState,
                                            reserveSpaceSize,
                                            dropoutStateSize
                                            };
  return per_device_state;
}

void forward_kernel(cudaStream_t stream,
                    DropoutPerDeviceState *m,
                    float const *input_ptr,
                    float *output_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnDropoutForward(m->handle.dnn,
                                 m->dropoutDesc,
                                 m->inputTensor,
                                 input_ptr,
                                 m->outputTensor,
                                 output_ptr,
                                 m->reserveSpace,
                                 m->reserveSpaceSize));
}

void backward_kernel(cudaStream_t stream,
                     DropoutPerDeviceState *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  checkCUDNN(cudnnDropoutBackward(m->handle.dnn,
                                  m->dropoutDesc,
                                  m->outputTensor,
                                  output_grad_ptr,
                                  m->inputTensor,
                                  input_grad_ptr,
                                  m->reserveSpace,
                                  m->reserveSpaceSize));
}

void cleanup_kernel(Realm::RegionInstance reserveInst,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffDropoutDescriptor_t dropoutDesc) {
  reserveInst.destroy();
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc));
}

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow
