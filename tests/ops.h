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

#ifndef _LEGION_CNN_OPS_H_
#define _LEGION_CNN_OPS_H_

//#define DISABLE_COMPUTATION
#include "legion.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <unistd.h>
#include "hdf5.h"
using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

#define MAX_NUM_INPUTS 512
#define MAX_NUM_LOCALS 3
#define MAX_NUM_WORKERS 16
#define MAX_DIM 4
#define MAX_FILENAME 200

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  CNN_INIT_TASK_ID,
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

enum Pool2DType {
  POOL2D_MAX,
  POOL2D_AVG,
};

enum FieldIDs {
  FID_DATA,
};

struct CnnHandle {
#ifndef DISABLE_COMPUTATION
  cudnnHandle_t dnn;
  cublasHandle_t blas;
#endif
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
  LogicalRegion region, region_grad;
  LogicalPartition partition, partition_grad;
};

struct TensorWithGrad {
  //int dim[MAX_DIM];
  LogicalRegion region, region_grad;
  LogicalPartition partition, partition_grad;
};

struct CnnConfig {
  Context lg_ctx;
  HighLevelRuntime *lg_hlr;
  FieldSpace field_space;
  //int num_par_h, num_par_w, num_par_n, num_workers;
  //int fc_num_par_c, fc_num_par_n;
  int sm_num_par, num_loaders, num_nodes;
  bool profiling;
  float learning_rate;
};

class OpMeta {
public:
  OpMeta(CnnHandle _handle) : handle(_handle) {};
public:
  CnnHandle handle;
};

class CnnModel;
class DataLoader;

class Op {
public:
  Op(Tensor input);
  Op(int num, Tensor* inputs);

  static void dummy_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime);

  virtual void prefetch(const CnnModel&);

  virtual void init(const CnnModel&) = 0;

  virtual void forward(const CnnModel&) = 0;

  virtual void backward(const CnnModel&) = 0;

  virtual void update(const CnnModel&) = 0;
public:
  Tensor output;
  //Op* pre_ops[MAX_NUM_INPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS];
  TensorWithGrad locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numLocals;
  //std::vector<LogicalRegion> inputs, grads;
};

class CnnModel {
public:
  CnnModel(int num_images, int height, int width,
           int image_par, int height_par, int width_par,
           int fc_par_n, int fc_par_c, bool profiling,
           float learning_rate,
           int num_loaders_per_node, int num_nodes,
           Context ctx, Runtime* runtime);

  static void init_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  void init_images();

  static void init_labels_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime); 
  void init_labels();

  void init_layers()
  {
    init_images();
    init_labels();
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->init(*this);
    }
  }

  static void load_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime);

  static void normalize_images_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
			            Context ctx, Runtime *runtime);

  void load_images();

  void prefetch();

  void forward();

  void backward();

  void update();

  Tensor add_conv_layer(Tensor input, int out_channels, int kernel_x, int kernel_y,
                        int stride_x, int stride_y, int padding_x, int padding_y, bool relu = true);

  Tensor add_pool_layer(Tensor input, int kernel_h, int kernel_w,
                        int stride_h, int stride_w, int padding_h, int padding_w,
                        Pool2DType type = POOL2D_MAX, bool relu = false);

  Tensor add_bn_layer(Tensor input, bool relu = true);

  Tensor add_linear_layer(Tensor input, int output_channels, bool relu = true);

  Tensor add_concat_layer(int n, Tensor* tensors);

  Tensor add_flat_layer(Tensor input);

  Tensor add_softmax_layer(Tensor input);
public:
  IndexSpaceT<3> part_is;
  IndexSpaceT<2> fc_part_is;
  IndexSpaceT<1> sm_part_is;
  IndexSpaceT<1> load_part_is;
  Tensor input_image, input_label;
  CnnConfig config;
  std::vector<Op*> layers;
  CnnHandle cnn_handlers[MAX_NUM_WORKERS];
  DataLoader *dataLoader;
  // regions/partitions for loading input images
  LogicalRegion rgb_lr;
  LogicalPartition rgb_image_lp, rgb_load_lp;
};

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

class Conv2D : public Op {
public:
  Conv2D(CnnConfig config, Tensor input, IndexSpaceT<4> part_is,
         int in_channels, int out_channels, int kernel_x, int kernel_y,
         int stride_x, int stride_y, int padding_x, int padding_y,
         bool relu, bool first_layer);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  int in_channels, out_channels;
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  bool relu, first_layer, profiling_runtime;
  int num_replica;
  float learning_rate;
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(CnnHandle handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
#endif
  bool relu, first_layer;
};

class Pooling2D : public Op {
public:
  Pooling2D(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
            int kernel_h, int kernel_w, int stride_h, int stride_w,
            int padding_h, int padding_w, Pool2DType type, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  Pool2DType pool_type;
  bool relu, profiling_runtime;
};

class Pooling2DMeta : public OpMeta {
public:
  Pooling2DMeta(CnnHandle handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#endif
  bool relu;
};

class BatchNorm : public Op {
public:
  BatchNorm(CnnConfig config, Tensor input, IndexSpaceT<3> part_is, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  bool relu, profiling_runtime;
  int num_replica;
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(CnnHandle handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
#endif
  bool relu;
};

class Linear : public Op {
public:
  Linear(CnnConfig config, Tensor input, IndexSpaceT<2> part_is,
         int output_channels, bool relu);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  bool relu, profiling_runtime;
  int in_channels, out_channels, num_replica, fc_num_par_c;
  float learning_rate;
};

class LinearMeta : public OpMeta {
public:
  LinearMeta(CnnHandle handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  int input_channels, output_channels, batch_size;
  bool relu;
  float *one_ptr, *pre_relu;
};

class Flat : public Op {
public:
  Flat(CnnConfig config, Tensor input,
       IndexSpaceT<3> part_is_3d,
       IndexSpaceT<2> part_is_2d);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  FlatMeta(CnnHandle handle) : OpMeta(handle) {};
};

class Softmax : public Op {
public:
  Softmax(CnnConfig config, Tensor input,
          IndexSpaceT<1> part_is);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  SoftmaxMeta(CnnHandle handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
#endif
};

class Concat : public Op {
public:
  Concat(CnnConfig config, int n, Tensor* inputs,
         IndexSpaceT<3> part_is);

  void init(const CnnModel&);

  void forward(const CnnModel&);

  void backward(const CnnModel&);

  void update(const CnnModel&);

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
  int num_inputs;
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(CnnHandle handle) : OpMeta(handle) {};
};

// class DataLoader

struct HDFFile {
  char filename[MAX_FILENAME];
  hid_t fid;
  hsize_t numImages, start, end;
};

struct DataLoadMeta {
  int cnt;
  HDFFile datasets[2];
};

class DataLoader {
public:
  DataLoader(std::string filename);
  void get_images(int numImages, DataLoadMeta &meta);
public:
  std::vector<HDFFile> datasets;
  int fileIdx, imageIdx;
};

#endif // _LEGION_CNN_OPS_H_
