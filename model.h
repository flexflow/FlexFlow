/* Copyright 2018 Stanford
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

#ifndef _FLEXFLOW_RUNTIME_H_
#define _FLEXFLOW_RUNTIME_H_
#include "legion.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <unistd.h>

// define constraints
#define MAX_NUM_INPUTS 6
#define MAX_NUM_LOCALS 3
#define MAX_NUM_WORKERS 16
#define MAX_DIM
#define MAX_FILENAME 200
#define MAX_OPNAME 64

using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
  BATCHNORM_INIT_TASK_ID,
  BATCHNORM_INIT_PARA_TASK_ID,
  BATCHNORM_FWD_TASK_ID,
  BATCHNORM_BWD_TASK_ID,
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_BWD2_TASK_ID,
  LINEAR_UPD_TASK_ID,
  FLAT_INIT_TASK_ID,
  FLAT_FWD_TASK_ID,
  FLAT_BWD_TASK_ID,
  SOFTMAX_INIT_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  DUMMY_TASK_ID,
};

enum PoolType {
  POOL_MAX,
  POOL_AVG,
};

enum FieldIDs {
  FID_DATA,
};

struct FFHandler {
  cudnnHandle_t cudnn;
  cublasHandle_t cublas;
  void *workSpace;
  size_t workSpaceSize;
};

struct Tensor {
  int numDim, adim[MAX_DIM], pdim[MAX_DIM];
  LogicalRegion region, regionGrad,
  LogicalPartition part, partGrad;
};

class OpMeta {
public:
  OpMeta(FFHandler _handle) : handle(_handle) {};
public:
  FFhandler handler;
};

class Op {
public:
  Op(std::string _name, Tensor input);
  Op(std::string _name, int num, Tensor* inputs);

  virtual void prefetch(const FFModel&);
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  virtual void update(const FFModel&) = 0;
public:
  char* name[MAX_OPNAME];
  Tensor output;
  Tensor inputs[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS];
  Tensor locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numLocals;
};

class FFModel {
public:
  FFModel(FFConfig config, Context ctx, Runtime* runtime);

  // Add a 2D convolutional layer 
  Tensor conv2d(std::string name,
                Tensor input, int outChannels,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                bool relu = false);
  // Add a 2D pooling layer
  Tensor pool2d(std::string name,
                Tensor input,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                PoolType type = POOL_MAX, bool relu = true);
  // Add a linear layer
  Tensor linear(std::string name,
                Tensor input, int outChannels,
                bool relu = true);
  // Add a concat layer
  Tensor concat(std::string name,
                int n, Tensor* tensors);
  // Add a flat layer
  Tensor flat(std::string name, Tensor input);
  // Add a softmax layer
  Tensor softmax(std::string name, Tensor input);

  void init_layers();
  void load_images();
  void prefetch();
  void forward();
  void backward();
  void update();
public:
  FFConfig config;
  std::vector<Op*> layers;
  FFHandlers handlers[MAX_NUM_WORKERS];
};

class Conv2D : public Op {
public:
  Conv2D(std::string name, FFConfig config,
         Tensor input, IndexSpaceT<3> part_is,
         int inChannels, int outChannels,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         bool relu, bool first_layer);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void init_para_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);

  static void update_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);
public:
  int inChannels, outChannels;
  int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
  bool relu, firstLayer, profiling;
  int numReplica;
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu, firstLayer;
};

class Pool2D : public Op {
public:
  Pool2D(std::string name, FFConfig config,
         Tensor input, IndexSpaceT<3> part_is,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, bool relu);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

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
  int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
  Pool2DType poolType;
  bool relu, profiling;
};

class Pool2DMeta : public OpMeta {
public:
  PoolDMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
};

class BatchNorm : public Op {
public:
  BatchNorm(std::string name, FFConfig config,
            Tensor input, IndexSpaceT<3> part_is,
            bool relu);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void init_para_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  bool relu, profiling;
  int numReplica;
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};

class Linear : public Op {
public:
  Linear(std::string name, FFConfig config,
         Tensor input, IndexSpaceT<2> part_is,
         int outChannels, bool relu);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void init_para_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);

  static void backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);

  static void update_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime);
public:
  LogicalPartition replica_sub_lps[MAX_NUM_WORKERS];
  bool relu, profiling;
  int inChannels, outChannels, numReplica, fcNumParC;
  float learning_rate;
};

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  int inChannels, outChannels, batchSize;
  bool relu;
  float *onePtr, *preRelu;
};

class Flat : public Op {
public:
  Flat(std::string name, FFConfig config,
       Tensor input,
       IndexSpaceT<3> part_is_3d,
       IndexSpaceT<2> part_is_2d);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

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
  LogicalPartition flat_lp, flat_grad_lp;
};

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Softmax : public Op {
public:
  Softmax(std::string name, FFConfig config,
          Tensor input, IndexSpaceT<1> part_is);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

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
  LogicalPartition input_grad_lp;
  bool profiling_runtime;
};

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
#endif
};

class Concat : public Op {
public:
  Concat(std::string name, FFConfig config,
         int n, Tensor* inputs, IndexSpaceT<3> part_is);

  void init(const FFModel&);

  void forward(const FFModel&);

  void backward(const FFModel&);

  void update(const FFModel&);

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
  int numInputs;
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
};

class UtilityTasks {
  FFHandler init_cuda_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void dummy_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime);
  static void init_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void load_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void normalize_images_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
};
#endif//_FLEXFLOW_RUNTIME_H_

