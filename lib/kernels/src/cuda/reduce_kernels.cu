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

#include "kernels/cuda_helper.h"
#include "kernels/reduce_kernels.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

ReducePerDeviceState::ReducePerDeviceState(FFHandler handler,
                                           Reduce const *rd,
                                           Domain const &input_domain)
    : op_type(rd->op_type), PerDeviceOpState(handler) {
  checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
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
  checkCUDNN(cudnnSetReduceTensorDescriptor(reduceDesc,
                                            reduce_op,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN,
                                            CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));
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
  checkCUDNN(cudnnDestroyReduceTensorDescriptor(reduceDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
}

namespace Kernels {
namespace Reduce {

ReducePerDeviceState init_kernel(PerDeviceFFhandle const & handle,
    ffTensorDescriptor_t const & input_tensor,
    ffTensorDescriptor_t const & outputTensor,
    ffReduceTensorDescriptor_t const & reduceDesc,
    OperatorType const & op_type,
    size_t const &  reduction_size) {
      return {handle, input_tensor, outputTensor, reduceDesc, op_type, reduction_size};
}

void forward_kernel(cudaStream_t stream,
                    ReducePerDeviceState const *m,
                    float const *input_ptr,
                    float *output_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnReduceTensor(m->handle.dnn,
                               m->reduceDesc,
                               nullptr /*indices*/,
                               0 /*indicesSizeInBytes*/,
                               m->handle.workSpace,
                               m->handle.workSpaceSize,
                               &alpha,
                               m->inputTensor,
                               input_ptr,
                               &beta,
                               m->outputTensor,
                               output_ptr));
};

void backward_kernel(cudaStream_t stream,
                     ReducePerDeviceState const *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha = 1.0, beta = 1.0f;
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
  checkCUDNN(cudnnAddTensor(m->handle.dnn,
                            &alpha,
                            m->outputTensor,
                            output_grad_ptr,
                            &beta,
                            m->inputTensor,
                            input_grad_ptr));
}

} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow
