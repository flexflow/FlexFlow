/* Copyright 2017 Stanford, NVIDIA
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

#include "legion.h"

#ifndef _LEGION_CNN_OPS_H_
#define _LEGION_CNN_OPS_H_

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

using namespace LegionRuntime::HighLevel;
#define MAX_NUM_INPUTS 3
#define MAX_NUM_LOCALS 2
#define MAX_DIM 4

struct CnnHandle {
  cudnnHandle_t dnn;
  cublasHandle_t blas;
} CnnHandle;

struct Tensor {
  int dim[MAX_DIM];
  LogicalRegion region;
} Tensor;

//struct TensorWithGrad {
//  int dim[MAX_DIM];
//  LogicalRegion region, region_grad;
//} TensorWithGrad;

class CnnContext {
public:
  CnnContext() {};
  ~CnnContext() {
    layers.clear();
  };
  Cnnhandle handle;
  std::vector<Op*> layers;
};

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime);

class Op {
public:
  Op(std::vector<Op*> previous_layers);

  virtual void init(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) = 0;

  virtual void forward(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) = 0;

  virtual void backward(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime) = 0;

public:
  Tensor output;
  Op* pre_ops[MAX_NUM_INPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  Tensor locals[MAX_NUM_LOCALS];
  //std::vector<LogicalRegion> inputs, grads;
};

class Conv2D : public Op {
public:
  Conv2D(int in_channels, int out_channels, int kernel_x, int kernel_y,
         int stride_x, int stride_y, int padding_x, int padding_y,
         Op* pre_op);

  void init(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime);

  void forward(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime);

  void backward(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime);
public:
  int in_channels, out_channels, kernel[2], stride[2], padding[2];
};

class Conv2DMeta : public OpMeta {
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
};
#endif // _LEGION_CNN_OPS_H_
