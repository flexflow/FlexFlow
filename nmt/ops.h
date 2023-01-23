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

#ifndef _LEGION_CNN_OPS_H_
#define _LEGION_CNN_OPS_H_

// #define DISABLE_COMPUTATION
#include "legion.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <unistd.h>
using namespace Legion;

template <typename FT, int N, typename T = coord_t>
using AccessorRO =
    FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = coord_t>
using AccessorRW =
    FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = coord_t>
using AccessorWO =
    FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define MAX_NUM_LOCALS 3
#define MAX_NUM_WORKERS 16
#define MAX_NUM_PARTS 16
#define MAX_DIM 4
#define MAX_FILENAME 200

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  CUDNN_INIT_TASK_ID,
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
  // RNN Task IDs
  LSTM_INIT_TASK_ID,
  LSTM_FWD_TASK_ID,
  LSTM_BWD_TASK_ID,
  RNN_LINEAR_INIT_TASK_ID,
  RNN_LINEAR_FWD_TASK_ID,
  RNN_LINEAR_BWD_TASK_ID,
  RNN_LINEAR_BWD2_TASK_ID,
  EMBED_INIT_TASK_ID,
  EMBED_FWD_TASK_ID,
  EMBED_BWD_TASK_ID,
  RNN_SOFTMAXDP_INIT_TASK_ID,
  RNN_SOFTMAXDP_FWD_TASK_ID,
  RNN_SOFTMAXDP_BWD_TASK_ID,
  PARAMS_INIT_TASK_ID,
  PARAMS_UPD_TASK_ID,
  WORD_INIT_TASK_ID, // DUMMY_TASK_ID: To be removed
  ZERO_1D_INIT_TASK_ID,
  ZERO_2D_INIT_TASK_ID,
  ZERO_3D_INIT_TASK_ID,
  // Dummy task ID
  DUMMY_TASK_ID,
};

enum Pool2DType {
  POOL2D_MAX,
  POOL2D_AVG,
};

enum FieldIDs {
  FID_DATA,
};

struct DnnHandle {
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
  // int dim[MAX_DIM];
  LogicalRegion region, region_grad;
  LogicalPartition partition, partition_grad;
};

class OpMeta {
public:
  OpMeta(DnnHandle _handle) : handle(_handle){};

public:
  DnnHandle handle;
};

// Empty base class
class CnnModel;
class DataLoader;

class Op {
public:
  Op(Tensor input);
  Op(int num, Tensor *inputs);
  virtual void init(CnnModel const &) = 0;

  virtual void forward(CnnModel const &) = 0;

  virtual void backward(CnnModel const &) = 0;

  virtual void update(CnnModel const &) = 0;

public:
  Tensor output;
  // Op* pre_ops[MAX_NUM_INPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS];
  TensorWithGrad locals[MAX_NUM_LOCALS];
  OpMeta *meta[MAX_NUM_WORKERS];
  // std::vector<LogicalRegion> inputs, grads;
};

DnnHandle init_cudnn(Task const *task,
                     std::vector<PhysicalRegion> const &regions,
                     Context ctx,
                     Runtime *runtime);

#endif // _LEGION_OPS_H_
