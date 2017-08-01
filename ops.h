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
#include <cudnn.h>
#include <cublas_v2.h>

#ifndef _LEGION_CNN_OPS_H_
#define _LEGION_CNN_OPS_H_

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
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

//using namespace LegionRuntime::HighLevel;
using namespace Legion;

#define MAX_NUM_INPUTS 3
#define MAX_NUM_LOCALS 2
#define MAX_NUM_WORKERS 16
#define MAX_DIM 4

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

struct CnnHandle {
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  void *workSpace;
  size_t workSpaceSize;
};

struct Tensor {
//  Tensor(int _numDim, int* _dim, LogicalRegion lr, LogicalPartition lp)
//  {
//    numDim = _numDim;
//    for (int i = 0; i < numDim; i++)
//      dim[i] = _dim[i];
//    region = lr;
//    partition = lp;
//  }
  int numDim, dim[MAX_DIM];
  LogicalRegion region;
  LogicalPartition partition;
};

struct TensorWithGrad {
  //int dim[MAX_DIM];
  LogicalRegion region, region_grad;
  LogicalPartition partition, partition_grad;
};

struct CnnConfig {
  Context lg_ctx;
  HighLevelRuntime *lg_hlr;
  int num_par_h, num_par_w, num_par_n, num_workers;
};

class OpMeta {
public:
  OpMeta() {};
  OpMeta(CnnHandle _handle) : handle(_handle) {};
public:
  CnnHandle handle;
};

class Op {
public:
  Op(Tensor input);

  virtual OpMeta* init(const Task *task,
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
  //Op* pre_ops[MAX_NUM_INPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  TensorWithGrad locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  //std::vector<LogicalRegion> inputs, grads;
};

class CnnModel {
public:
  CnnModel(int num_images, int height, int width,
           int image_par, int height_par, int width_par,
           Context ctx, Runtime* runtime)
  {
    config.lg_ctx = ctx;
    config.lg_hlr = runtime;
    config.num_par_w = width_par;
    config.num_par_h = height_par;
    config.num_par_n = image_par;
    Realm::ZRect<3, coord_t> image_rect(Realm::ZPoint<3>(0, 0, 0),
                                        Realm::ZPoint<3>(width-1, height-1, num_images*3-1));
    IndexSpaceT<3> image_is = runtime->create_index_space(ctx, image_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(float), FID_DATA);
    }
    LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is, fs);
  };

  Tensor add_conv_layer(Tensor input, int out_channels, int kernel_x, int kernel_y,
                        int stride_x, int stride_y, int padding_x, int padding_y);

  Tensor add_pooling_layer(Tensor input);
public:
  CnnConfig config;
  std::vector<Op*> layers;
};

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

class Conv2D : public Op {
public:
  Conv2D(CnnConfig config, Tensor input,
         int in_channels, int out_channels, int kernel_x, int kernel_y,
         int stride_x, int stride_y, int padding_x, int padding_y);

  OpMeta* init(const Task *task,
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
public:
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
};

/*
class Pooling2D : public Op {
public:
  Pooling2D(int kernel_h, int kernel_w, int stride_h = 0, int stride_w = 0,
          int padding_h = 0, int padding_w = 0);

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
  int kernel[2], stride[2], padding[2];
};
*/
#endif // _LEGION_CNN_OPS_H_
