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
  CNN_INIT_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
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
  int numDim, adim[MAX_DIM], pdim[MAX_DIM];
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
  OpMeta(CnnHandle _handle) : handle(_handle) {};
public:
  CnnHandle handle;
};

class CnnModel;

class Op {
public:
  Op(Tensor input);

  virtual void init(const CnnModel&) = 0;

  virtual void forward(const CnnModel&) = 0;

  virtual void backward(const CnnModel&) = 0;

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
           Context ctx, Runtime* runtime);

  void init_layers()
  {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->init(*this);
    }
  }

  void forward()
  {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->forward(*this);
    }
  }

  Tensor add_conv_layer(Tensor input, int out_channels, int kernel_x, int kernel_y,
                        int stride_x, int stride_y, int padding_x, int padding_y, bool relu);

  Tensor add_pooling_layer(Tensor input, int kernel_h, int kernel_w,
                           int stride_h, int stride_w, int padding_h, int padding_w, bool relu);

  Tensor add_linear_layer(Tensor input, int output_channels, bool relu);
public:
  IndexSpaceT<3> part_is;
  Tensor input_image;
  CnnConfig config;
  std::vector<Op*> layers;
  CnnHandle cnn_handlers[MAX_NUM_WORKERS];
};

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

class Conv2D : public Op {
public:
  Conv2D(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
         int in_channels, int out_channels, int kernel_x, int kernel_y,
         int stride_x, int stride_y, int padding_x, int padding_y, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
public:
  int in_channels, out_channels;
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(CnnHandle handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
};

class Pooling2D : public Op {
public:
  Pooling2D(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
            int kernel_h, int kernel_w, int stride_h, int stride_w,
            int padding_h, int padding_w, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);

public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
};

class Pooling2DMeta : public OpMeta {
public:
  Pooling2DMeta(CnnHandle handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnPoolingDescriptor_t poolDesc;
};

class Linear : public Op {
public:
  Linear(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
         int output_channels, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);

};

class LinearMeta : public OpMeta {
public:
  LinearMeta(Cnnhandle handle) : OpMeta(handle) {};
};

class Flat : public Op {
public:
  Flat(CnnConfig config, Tensor input);
};
#endif // _LEGION_CNN_OPS_H_
