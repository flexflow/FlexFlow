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
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;
using Legion::Memory;

DropoutPerDeviceState::DropoutPerDeviceState(FFHandler handler,
                                             bool profiling,
                                             float rate,
                                             unsigned long long seed,
                                             Memory gpu_mem,
                                             Domain const &output_domain)
    : PerDeviceOpState(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(miopenDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, output_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_domain));
  checkCUDNN(
      miopenDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                   Realm::Point<1, coord_t>(totalSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    dropoutStates = reserveInst.pointer_untyped(0, sizeof(char));
    reserveSpace = ((char *)dropoutStates) + dropoutStateSize;
  }
  // checkCUDA(hipMalloc(&dropoutStates, dropoutStateSize));
  // checkCUDA(hipMalloc(&reserveSpace, reserveSpaceSize));
  checkCUDNN(miopenSetDropoutDescriptor(dropoutDesc,
                                        handle.dnn,
                                        rate,
                                        dropoutStates,
                                        dropoutStateSize,
                                        seed,
                                        false,
                                        false,
                                        MIOPEN_RNG_PSEUDO_XORWOW));
}

DropoutPerDeviceState::~DropoutPerDeviceState(void) {
  reserveInst.destroy();
  checkCUDNN(miopenDestroyTensorDescriptor(inputTensor));
  checkCUDNN(miopenDestroyTensorDescriptor(outputTensor));
  checkCUDNN(miopenDestroyDropoutDescriptor(dropoutDesc));
}

namespace Kernels {
namespace Dropout {

void forward_kernel(hipStream_t stream,
                    DropoutPerDeviceState *m,
                    float const *input_ptr,
                    float *output_ptr) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  checkCUDNN(miopenDropoutForward(m->handle.dnn,
                                  m->dropoutDesc,
                                  m->inputTensor /* not used */,
                                  m->inputTensor,
                                  input_ptr,
                                  m->outputTensor,
                                  output_ptr,
                                  m->reserveSpace,
                                  m->reserveSpaceSize));
}

void backward_kernel(hipStream_t stream,
                     DropoutPerDeviceState *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  checkCUDNN(miopenDropoutBackward(m->handle.dnn,
                                   m->dropoutDesc,
                                   m->inputTensor /* not used */,
                                   m->outputTensor,
                                   output_grad_ptr,
                                   m->inputTensor,
                                   input_grad_ptr,
                                   m->reserveSpace,
                                   m->reserveSpaceSize));
}

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow
