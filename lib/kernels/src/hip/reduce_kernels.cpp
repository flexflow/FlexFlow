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

#include "kernels/reduce_kernels.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

ReducePerDeviceState::ReducePerDeviceState(FFHandler handler, Reduce const *rd,
                                           Domain const &input_domain)
    : op_type(rd->op_type), PerDeviceOpState(handler) {
  checkCUDNN(miopenCreateReduceTensorDescriptor(&reduceDesc));
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  cudnnReduceTensorOp_t reduce_op;
  switch (rd->op_type) {
  case OP_REDUCE_SUM:
    reduce_op = CUDNN_REDUCE_TENSOR_ADD;
    break;
  case OP_REDUCE_MEAN:
    reduce_op = CUDNN_REDUCE_TENSOR_AVG;
    break;
  default:
    assert(false);
  }
  checkCUDNN(miopenSetReduceTensorDescriptor(
      reduceDesc, MIOPEN_REDUCE_TENSOR_ADD, miopenFloat, MIOPEN_PROPAGATE_NAN,
      MIOPEN_REDUCE_TENSOR_NO_INDICES, MIOPEN_32BIT_INDICES));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_domain));
  Domain output_domain = input_domain;
  for (size_t i = 0; i < rd->num_axes; i++) {
    assert(input_domain.dim > rd->axes[i]);
    output_domain.rect_data[rd->axes[i] + output_domain.dim] =
        output_domain.rect_data[rd->axes[i]];
  }
  assert(output_domain.get_volume() % input_domain.get_volume() == 0);
  reduction_size = input_domain.get_volume() / output_domain.get_volume();
  assert(reduction_size > 0);
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_domain));
}

ReducePerDeviceState::~ReducePerDeviceState(void) {
  checkCUDNN(miopenDestroyReduceTensorDescriptor(reduceDesc));
  checkCUDNN(miopenDestroyTensorDescriptor(inputTensor));
  checkCUDNN(miopenDestroyTensorDescriptor(outputTensor));
}

namespace Kernels {
namespace Reduce {

void forward_kernel(hipStream_t stream, ReducePerDeviceState const *m,
                    float const *input_ptr, float *output_ptr) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenReduceTensor(
      m->handle.dnn, m->reduceDesc, nullptr /*indices*/,
      0 /*indicesSizeInBytes*/, m->handle.workSpace, m->handle.workSpaceSize,
      &alpha, m->inputTensor, input_ptr, &beta, m->outputTensor, output_ptr));
};

void backward_kernel(hipStream_t stream, ReducePerDeviceState const *m,
                     float const *output_grad_ptr, float *input_grad_ptr) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  switch (m->op_type) {
  case OP_REDUCE_SUM:
    alpha = 1.0f;
    break;
  case OP_REDUCE_MEAN:
    // When the output is the average of multiple input elements
    // we need to scale the gradients by 1.0 / reduction_size
    alpha = 1.0f / m->reduction_size;
    break;
  default:
    assert(false);
  }
  checkCUDNN(miopenOpTensor(m->handle.dnn, miopenTensorOpAdd, &alpha,
                            m->inputTensor, input_grad_ptr, &alpha,
                            m->outputTensor, output_grad_ptr, &beta,
                            m->inputTensor, input_grad_ptr));
}

} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow
