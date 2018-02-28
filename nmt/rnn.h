/* Copyright 2018 Stanford, NVIDIA
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

#ifndef _LEGION_RNN_H_
#define _LEGION_RNN_H_

#include "ops.h"

#define MAX_SEQ_LENGTH 40
#define MAX_NUM_LAYERS 4
#define LSTM_PER_NODE_LENGTH 5

struct RnnConfig {
  Context lg_ctx;
  HighLevelRuntime *lg_hlr;
  FieldSpace field_space;
  int batchSize, hiddenSize, embedSize;
  int numLayers, seqLength, numParts;
  int numNodes, workersPerNode;
};

struct SharedVariable {
  LogicalRegion region, gradients[MAX_NUM_WORKERS];
  LogicalRegion subregions[2*MAX_NUM_PARTS];
  SharedVariable() {
    region = LogicalRegion::NO_REGION;
    for (int i = 0; i < MAX_NUM_WORKERS; i++)
      gradients[i] = LogicalRegion::NO_REGION;
    for (int i = 0; i < 2*MAX_NUM_PARTS; i++)
      subregions[i] = LogicalRegion::NO_REGION;
  }
};

struct ParallelConfig {
  int nDims, dim[MAX_DIM];
  int gpu[MAX_NUM_WORKERS];
};

struct GlobalConfig {
  ParallelConfig linear[MAX_SEQ_LENGTH];
  ParallelConfig lstm[MAX_NUM_LAYERS][2*MAX_SEQ_LENGTH];
  ParallelConfig embed[2*MAX_SEQ_LENGTH];
};

class RnnModel;

class RnnOp {
public:
  RnnOp(Tensor input, ParallelConfig pc, SharedVariable _params);
  RnnOp(Tensor t1, Tensor t2, Tensor t3, ParallelConfig pc, SharedVariable _params);
  RnnOp(int num, Tensor* inputs);
  virtual void init(const RnnModel&) = 0;

  virtual void forward(const RnnModel&) = 0;

  virtual void backward(const RnnModel&) = 0;

  virtual void update(const RnnModel&) = 0;
public:
  Tensor outputs[MAX_NUM_OUTPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  OpMeta* meta[MAX_NUM_WORKERS];
  ParallelConfig paraConfig;
  SharedVariable params;
};

struct LSTMTensors {
  Tensor x, hx, cx;
};

class RnnModel {
public:
  RnnModel(int batch_size, int numLayers, int seqLength,
           int hidden_size, int embed_size, int vocab_size,
           int num_parts, int num_nodes, int num_workers_per_node,
           GlobalConfig global,
           Context ctx, Runtime *runtime);

  void init();

  void forward();

  void backward();

  void update();

  void update_shared_variable(SharedVariable params);

  static void params_update_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, HighLevelRuntime *runtime);

  LSTMTensors add_lstm_node(Tensor x, Tensor hx, Tensor cx,
                            ParallelConfig pc, SharedVariable params);

  Tensor add_linear_node(Tensor x, int output_size,
                         ParallelConfig pc, SharedVariable params);
public:
  RnnConfig config;
  std::vector<RnnOp*> layers;
  std::vector<SharedVariable> sharedVariables;
  DnnHandle dnn_handlers[MAX_NUM_WORKERS];
  Tensor srcs[MAX_SEQ_LENGTH], dsts[MAX_SEQ_LENGTH];
  IndexSpaceT<1> part_is;
};

/*
 * For now, every single LSTM cell with 1 word and 1 layer is a
 * LSTM operation.
 */
class LSTM : public RnnOp {
public:
  LSTM(RnnConfig config, Tensor x, Tensor hx, Tensor cx,
       int batch_size, int input_size, int output_size,
       ParallelConfig pc, SharedVariable params);

  void init(const RnnModel&);

  void forward(const RnnModel&);

  void backward(const RnnModel&);

  void update(const RnnModel&);

  static OpMeta* init_task(const Task *task,
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
  int batch_size, input_size, output_size;
  Rect<1> part_rect;
};

class LSTMMeta : public OpMeta {
public:
  LSTMMeta(DnnHandle handle) : OpMeta(handle) {};
  cudnnRNNDescriptor_t rnnDesc;
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t xDescs[LSTM_PER_NODE_LENGTH],
                          yDescs[LSTM_PER_NODE_LENGTH],
                          cxDesc, hxDesc, cyDesc, hyDesc;
  cudnnFilterDescriptor_t wDesc;
  size_t reserveSpaceSize;
  void* reserveSpace;
  bool profiling_runtime;
};

class Linear : public RnnOp {
public:
  Linear(RnnConfig config, Tensor input, int output_channels,
         ParallelConfig pc, SharedVariable params,
         IndexSpaceT<1> input_part_is);

  void init(const RnnModel&);

  void forward(const RnnModel&);

  void backward(const RnnModel&);

  void update(const RnnModel &);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);

  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);

  static void backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);

  static void update_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);
public:
  int batch_size, input_size, output_size;
  Tensor replica;
  // each replica_sub_lps[i] is a disjoint partition
  LogicalPartition replica_sub_lps[MAX_NUM_WORKERS];
  // input_lp may be an aliased partition if num_par_c > 1
  LogicalPartition input_lp;
  Rect<2> part_rect;
  Rect<1> input_part_rect;
};

class LinearMeta : public OpMeta {
public:
  LinearMeta(DnnHandle handle) : OpMeta(handle) {};
  float* one_ptr;
  bool profiling_runtime;
};

#endif //_LEGION_RNN_H_
