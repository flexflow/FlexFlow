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

#include "flexflow/ops/kernels/element_binary_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

ElementBinaryMeta::ElementBinaryMeta(FFHandler handler, Op const *op)
    : OpMeta(handler, op) {
  checkCUDNN(cudnnCreateTensorDescriptor(&input1Tensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&input2Tensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceAddDesc));
  op_type = OP_NOOP;
  profiling = false;
  inference_debugging = false;
  inplace_a = false;
  has_same_operands = false;
  broadcast_input1 = false;
  broadcast_input2 = false;
}

namespace Kernels {
namespace ElementBinary {

/*static*/
void init_kernel(ElementBinaryMeta *m,
                 Domain const &input1_domain,
                 Domain const &input2_domain,
                 Domain const &output_domain) {
  cudnnOpTensorOp_t mode;
  switch (m->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    case OP_EW_MAX:
      mode = CUDNN_OP_TENSOR_MAX;
      break;
    case OP_EW_MIN:
      mode = CUDNN_OP_TENSOR_MIN;
      break;
    default:
      assert(false);
  }
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  checkCUDNN(cudnnSetOpTensorDescriptor(
      m->opDesc, mode, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  checkCUDNN(cudnnSetReduceTensorDescriptor(m->reduceAddDesc,
                                            CUDNN_REDUCE_TENSOR_ADD,
                                            cudnn_data_type,
                                            CUDNN_PROPAGATE_NAN,
                                            CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(
      m->input1Tensor, input1_domain, m->input_type[0]));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(
      m->input2Tensor, input2_domain, m->input_type[1]));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(
      m->outputTensor, output_domain, m->output_type[0]));
}

/*static*/
void forward_kernel_wrapper(ElementBinaryMeta const *m,
                            GenericTensorAccessorR const &in1,
                            GenericTensorAccessorR const &in2,
                            GenericTensorAccessorW const &out) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  assert(in1.data_type == in2.data_type);
  assert(out.data_type == in1.data_type);
  if (out.data_type == DT_HALF) {
    Internal::forward_kernel(
        m, in1.get_half_ptr(), in2.get_half_ptr(), out.get_half_ptr(), stream);
  } else if (out.data_type == DT_FLOAT) {
    Internal::forward_kernel(m,
                             in1.get_float_ptr(),
                             in2.get_float_ptr(),
                             out.get_float_ptr(),
                             stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    char const *opName;
    switch (m->op_type) {
      case OP_EW_ADD:
        opName = "Add";
        break;
      case OP_EW_SUB:
        opName = "Sub";
        break;
      case OP_EW_MUL:
        opName = "Mul";
        break;
      case OP_EW_DIV:
        opName = "Div";
        break;
      case OP_EW_MAX:
        opName = "Max";
        break;
      case OP_EW_MIN:
        opName = "Min";
        break;
      default:
        assert(false);
    }
    printf("[%s] forward time (CF) = %.9fms\n", opName, elapsed);
    // print_tensor<float>(in1_ptr, 32, "[EWB:forward:input1]");
    // print_tensor<float>(in2_ptr, 32, "[EWB:forward:input2]");
    // print_tensor<float>(out_ptr, 32, "[EWB:forward:output]");
  }
}

/*static*/
void backward_kernel_wrapper(ElementBinaryMeta const *m,
                             float const *out_grad_ptr,
                             float const *in1_ptr,
                             float const *in2_ptr,
                             float *in1_grad_ptr,
                             float *in2_grad_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Internal::backward_kernel(
      m, out_grad_ptr, in1_ptr, in2_ptr, in1_grad_ptr, in2_grad_ptr, stream);
  // elewise_binary_backward_kernel<<<GET_BLOCKS(out_grad_domain.get_volume()),
  // CUDA_NUM_THREADS>>>( out_grad_domain.get_volume(), alpha, alpha,
  // ele->op_type, out_grad_ptr, in1_ptr, in2_ptr, in1_grad_ptr, in2_grad_ptr);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    char const *opName;
    switch (m->op_type) {
      case OP_EW_ADD:
        opName = "Add";
        break;
      case OP_EW_SUB:
        opName = "Sub";
        break;
      case OP_EW_MUL:
        opName = "Mul";
        break;
      case OP_EW_DIV:
        opName = "Div";
        break;
      case OP_EW_MAX:
        opName = "Max";
        break;
      case OP_EW_MIN:
        opName = "Min";
        break;
      default:
        assert(false);
    }
    printf("[%s] backward time (CB) = %.2fms\n", opName, elapsed);
  }
}

namespace Internal {

__global__ void elewise_binary_forward_kernel(coord_t volume,
                                              float const alpha,
                                              float const beta,
                                              OperatorType type,
                                              float const *in1,
                                              float const *in2,
                                              float *out) {
  switch (type) {
    case OP_EW_ADD: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_SUB: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_MUL: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * in1[i] * in2[i] + beta * out[i];
      }
      break;
    }
    case OP_EW_DIV: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] / in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_MAX: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * max(in1[i], in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_MIN: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * min(in1[i], in2[i]) + beta * out[i];
      }
      break;
    }
    default:
      assert(false);
  }
}

__global__ void elewise_binary_backward_kernel(coord_t volume,
                                               float const alpha,
                                               float const beta,
                                               OperatorType type,
                                               float const *out_grad,
                                               float const *in1,
                                               float const *in2,
                                               float *in1_grad,
                                               float *in2_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OP_EW_ADD: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_SUB: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = -alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_MUL: {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_DIV: {
        in1_grad[i] = alpha * out_grad[i] / in2[i] + beta * in1_grad[i];
        in2_grad[i] = -alpha * out_grad[i] * in1[i] / (in2[i] * in2[i]) +
                      beta * in2_grad[i];
        break;
      }
      case OP_EW_MAX: {
        in1_grad[i] = (in1[i] >= in2[i])
                          ? alpha * out_grad[i] + beta * in1_grad[i]
                          : beta * in1_grad[i];
        in2_grad[i] = (in2[i] >= in1[i])
                          ? alpha * out_grad[i] + beta * in2_grad[i]
                          : beta * in2_grad[i];
        break;
      }
      case OP_EW_MIN: {
        in1_grad[i] = (in1[i] <= in2[i])
                          ? alpha * out_grad[i] + beta * in1_grad[i]
                          : beta * in1_grad[i];
        in2_grad[i] = (in2[i] <= in1[i])
                          ? alpha * out_grad[i] + beta * in2_grad[i]
                          : beta * in2_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
template <typename DT>
void forward_kernel(ElementBinaryMeta const *m,
                    DT const *in1_ptr,
                    DT const *in2_ptr,
                    DT *out_ptr,
                    cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (m->op_type) {
    case OP_EW_SUB:
      alpha2 = -1.0f;
      break;
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_EW_MAX:
    case OP_EW_MIN:
      break;
    default:
      assert(false);
  }
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (m->broadcast_input1) {
    // currently only handle add and sub
    assert(m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD ||
           m->op_type == OP_EW_MUL);
    if (m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD) {
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(m->handle.dnn,
                               m->opDesc,
                               &beta,
                               m->outputTensor,
                               out_ptr,
                               &alpha1,
                               m->input1Tensor,
                               in1_ptr,
                               &beta,
                               m->outputTensor,
                               out_ptr));
      // output = (beta*output + alpha2*input2) + alpha1*output = alpha2*input2
      // + alpha1*input1
      checkCUDNN(cudnnOpTensor(m->handle.dnn,
                               m->opDesc,
                               &beta,
                               m->outputTensor,
                               out_ptr,
                               &alpha2,
                               m->input2Tensor,
                               in2_ptr,
                               &alpha1,
                               m->outputTensor,
                               out_ptr));
    } else if (m->op_type == OP_EW_MUL) {
      checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc,
                                            CUDNN_OP_TENSOR_ADD,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(m->handle.dnn,
                               m->opDesc,
                               &beta,
                               m->outputTensor,
                               out_ptr,
                               &alpha1,
                               m->input1Tensor,
                               in1_ptr,
                               &beta,
                               m->outputTensor,
                               out_ptr));
      checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc,
                                            CUDNN_OP_TENSOR_MUL,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (alpha1*output * alpha2*input2) + beta*output
      checkCUDNN(cudnnOpTensor(m->handle.dnn,
                               m->opDesc,
                               &alpha1,
                               m->outputTensor,
                               out_ptr,
                               &alpha2,
                               m->input2Tensor,
                               in2_ptr,
                               &beta,
                               m->outputTensor,
                               out_ptr));
    }
  } else {
    checkCUDNN(cudnnOpTensor(m->handle.dnn,
                             m->opDesc,
                             &alpha1,
                             m->input1Tensor,
                             in1_ptr,
                             &alpha2,
                             m->input2Tensor,
                             in2_ptr,
                             &beta,
                             m->outputTensor,
                             out_ptr));
  }
}

/*static*/
void backward_kernel(ElementBinaryMeta const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr,
                     cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  if (m->op_type == OP_EW_ADD || m->op_type == OP_EW_SUB) {
    float alpha = 1.0f, beta = 1.0f;
    if (in1_grad_ptr != nullptr) {
      if (m->broadcast_input1) {
        checkCUDNN(cudnnReduceTensor(m->handle.dnn,
                                     m->reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     m->handle.workSpace,
                                     m->handle.workSpaceSize,
                                     &alpha,
                                     m->outputTensor,
                                     out_grad_ptr,
                                     &beta,
                                     m->input1Tensor,
                                     in1_grad_ptr));
      } else {
        checkCUDNN(cudnnAddTensor(m->handle.dnn,
                                  &alpha,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m->input1Tensor,
                                  in1_grad_ptr));
      }
    }
    if (m->op_type == OP_EW_SUB) {
      alpha = -1.0f;
    }
    if (in2_grad_ptr != nullptr) {
      if (m->broadcast_input2) {
        checkCUDNN(cudnnReduceTensor(m->handle.dnn,
                                     m->reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     m->handle.workSpace,
                                     m->handle.workSpaceSize,
                                     &alpha,
                                     m->outputTensor,
                                     out_grad_ptr,
                                     &beta,
                                     m->input2Tensor,
                                     in2_grad_ptr));
      } else {
        checkCUDNN(cudnnAddTensor(m->handle.dnn,
                                  &alpha,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m->input2Tensor,
                                  in2_grad_ptr));
      }
    }
  } else if (m->op_type == OP_EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f, zero = 0.0f;
    if (in1_grad_ptr != nullptr) {
      if (m->broadcast_input1) {
        checkCUDNN(cudnnOpTensor(m->handle.dnn,
                                 m->opDesc,
                                 &alpha1,
                                 m->outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m->input2Tensor,
                                 in2_ptr,
                                 &zero,
                                 m->outputTensor,
                                 m->handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            m->handle.dnn,
            m->reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)m->handle.workSpace + sizeof(*out_grad_ptr)),
            m->handle.workSpaceSize - sizeof(*out_grad_ptr),
            &alpha1,
            m->outputTensor,
            m->handle.workSpace,
            &beta,
            m->input1Tensor,
            in1_grad_ptr));
      } else {
        checkCUDNN(cudnnOpTensor(m->handle.dnn,
                                 m->opDesc,
                                 &alpha1,
                                 m->outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m->input2Tensor,
                                 in2_ptr,
                                 &beta,
                                 m->input1Tensor,
                                 in1_grad_ptr));
      }
    }
    if (in2_grad_ptr != nullptr) {
      if (m->broadcast_input2) {
        checkCUDNN(cudnnOpTensor(m->handle.dnn,
                                 m->opDesc,
                                 &alpha1,
                                 m->outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m->input1Tensor,
                                 in1_ptr,
                                 &zero,
                                 m->outputTensor,
                                 m->handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            m->handle.dnn,
            m->reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)m->handle.workSpace + sizeof(*out_grad_ptr)),
            m->handle.workSpaceSize - sizeof(*out_grad_ptr),
            &alpha1,
            m->outputTensor,
            m->handle.workSpace,
            &beta,
            m->input2Tensor,
            in2_grad_ptr));
      } else {
        checkCUDNN(cudnnOpTensor(m->handle.dnn,
                                 m->opDesc,
                                 &alpha1,
                                 m->outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m->input1Tensor,
                                 in1_ptr,
                                 &beta,
                                 m->input2Tensor,
                                 in2_grad_ptr));
      }
    }
  } else if (m->op_type == OP_EW_MIN || m->op_type == OP_EW_MAX) {
    float alpha = 1.0f, beta = 1.0f;
    cudnnDataType_t dataType;
    int n;
    int dims[MAX_TENSOR_DIM];
    int strides[MAX_TENSOR_DIM];
    checkCUDNN(cudnnGetTensorNdDescriptor(
        m->outputTensor, MAX_TENSOR_DIM, &dataType, &n, dims, strides));
    size_t volume = 1;
    for (int i = 0; i < n; i++) {
      volume *= dims[i];
    }
    elewise_binary_backward_kernel<<<GET_BLOCKS(volume),
                                     CUDA_NUM_THREADS,
                                     0,
                                     stream>>>(volume,
                                               alpha,
                                               beta,
                                               m->op_type,
                                               out_grad_ptr,
                                               in1_ptr,
                                               in2_ptr,
                                               in1_grad_ptr,
                                               in2_grad_ptr);
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
}

} // namespace Internal
} // namespace ElementBinary
} // namespace Kernels
}; // namespace FlexFlow
