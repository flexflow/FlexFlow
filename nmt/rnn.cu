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

#include "rnn.h"
#include "rnn_mapper.h"
#include "../cnn_helper.h"

DnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 0);
  assert(task->arglen == sizeof(size_t));
  size_t workSpaceSize = *(const size_t*) task->args;
  DnnHandle handle;
  handle.workSpaceSize = workSpaceSize;
  printf("workSpaceSize = %zu\n", workSpaceSize);
#ifndef DISABLE_COMPUTATION
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
#endif
  checkCUDA(cudaMalloc(&handle.workSpace, workSpaceSize));
  return handle;
}

RnnOp::RnnOp(Tensor input, ParallelConfig pc, SharedVariable _params)
: paraConfig(pc), params(_params)
{
  inputs[0] = input;
}

RnnOp::RnnOp(Tensor t1, Tensor t2, Tensor t3, ParallelConfig pc, SharedVariable _params)
: paraConfig(pc), params(_params)
{
  inputs[0] = t1;
  inputs[1] = t2;
  inputs[2] = t3;
}

RnnOp::RnnOp(int n, Tensor *_inputs)
{
  for (int i = 0; i < n; i++) {
    inputs[i] = _inputs[i];
  }
}

RnnModel::RnnModel(int batch_size, int numLayers, int seqLength,
                   int hidden_size, int embed_size, int vocab_size,
                   int num_parts, int num_nodes, int num_gpus_per_node,
                   GlobalConfig global,
                   Context ctx, Runtime *runtime)
{
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.batchSize = batch_size;
  config.hiddenSize = hidden_size;
  config.embedSize = embed_size;
  config.numLayers = numLayers;
  config.seqLength = seqLength;
  config.numParts = num_parts;
  config.numNodes = num_nodes;
  config.workersPerNode = num_gpus_per_node;
  config.field_space = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  Rect<1> part_rect(Point<1>(0), Point<1>(num_parts-1));
  part_is = runtime->create_index_space(ctx, part_rect);
  assert(seqLength <= MAX_SEQ_LENGTH);
  assert(numLayers <= MAX_NUM_LAYERS);
  Rect<3> word_rect(Point<3>(0, 0, 0),
                    Point<3>(embed_size-1, batch_size-1, LSTM_PER_NODE_LENGTH-1));
  IndexSpaceT<3> word_is = runtime->create_index_space(ctx, word_rect);
  int extent_c = embed_size;
  int extent_n = batch_size / num_parts;
  Rect<3, coord_t> extent(Point<3>(0, 0, 0),
                          Point<3>(extent_c-1, extent_n-1, LSTM_PER_NODE_LENGTH-1));
  Transform<3, 1, coord_t> trans;
  trans[0][0] = 0; trans[1][0] = extent_n; trans[2][0] = 0;
  IndexPartition word_ip =
    runtime->create_partition_by_restriction(ctx, word_is, part_is, trans, extent);
  assert(runtime->is_index_partition_disjoint(ctx, word_ip));
  assert(runtime->is_index_partition_complete(ctx, word_ip));
  assert(seqLength % LSTM_PER_NODE_LENGTH == 0);
  int nodes_per_layer = seqLength / LSTM_PER_NODE_LENGTH;
  for (int i = 0; i < nodes_per_layer; i++) {
    srcs[i].numDim = 3;
    srcs[i].adim[0] = embed_size;
    srcs[i].adim[1] = batch_size;
    srcs[i].adim[2] = LSTM_PER_NODE_LENGTH;
    srcs[i].pdim[0] = extent_c;
    srcs[i].pdim[1] = extent_n;
    srcs[i].pdim[2] = LSTM_PER_NODE_LENGTH;
    srcs[i].region = runtime->create_logical_region(ctx, word_is, config.field_space);
    srcs[i].partition =
      runtime->get_logical_partition(ctx, srcs[i].region, word_ip);
    srcs[i].region_grad =
      runtime->create_logical_region(ctx, word_is, config.field_space);
    srcs[i].partition_grad =
      runtime->get_logical_partition(ctx, srcs[i].region_grad, word_ip);
    dsts[i] = srcs[i];
    dsts[i].region = runtime->create_logical_region(ctx, word_is, config.field_space);
    dsts[i].partition =
      runtime->get_logical_partition(ctx, dsts[i].region, word_ip);
    dsts[i].region_grad =
      runtime->create_logical_region(ctx, word_is, config.field_space);
    dsts[i].partition_grad =
      runtime->get_logical_partition(ctx, dsts[i].region_grad, word_ip);
  }
  // Create a zeroed tensor
  Rect<2> hx_rect(Point<2>(0, 0), Point<2>(hidden_size-1, batch_size-1));
  IndexSpaceT<2> hx_is = runtime->create_index_space(ctx, hx_rect);
  extent_c = hidden_size;
  extent_n = batch_size / num_parts;
  Rect<2> hx_ext(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  Transform<2, 1, coord_t> hx_trans;
  hx_trans[0][0] = 0; hx_trans[1][0] = extent_n;
  IndexPartition hx_ip =
    runtime->create_partition_by_restriction(ctx, hx_is, part_is, hx_trans, hx_ext);
  assert(runtime->is_index_partition_disjoint(ctx, hx_ip));
  assert(runtime->is_index_partition_complete(ctx, hx_ip));
  LSTMTensors zero[MAX_NUM_LAYERS], lstm[MAX_NUM_LAYERS][2*MAX_SEQ_LENGTH];
  for (int i = 0; i < numLayers; i++) {
    for (int j = 0; j < 2; j++) {
      Tensor t;
      t.numDim = 2;
      t.adim[0] = hidden_size;
      t.adim[1] = batch_size;
      t.pdim[0] = extent_c;
      t.pdim[1] = extent_n;
      t.region = runtime->create_logical_region(ctx, hx_is, config.field_space);
      t.partition = runtime->get_logical_partition(ctx, t.region, hx_ip);
      t.region_grad = runtime->create_logical_region(ctx, hx_is, config.field_space);
      t.partition_grad = runtime->get_logical_partition(ctx, t.region_grad, hx_ip);
      if (j == 0)
        zero[i].hx = t;
      else
        zero[i].cx = t;
    }
  }
  SharedVariable encoders[MAX_NUM_LAYERS], decoders[MAX_NUM_LAYERS];
  for (int i = 0; i < numLayers; i++) {
    int input_size = (i==0) ? embed_size : hidden_size;
    int output_size = hidden_size;
    int numParams = (input_size + 1 + output_size + 1) * output_size * 4;
    Rect<1> params_rect(Point<1>(0), Point<1>(numParams-1));
    IndexSpaceT<1> params_is = runtime->create_index_space(ctx, params_rect);
    encoders[i].region =
      runtime->create_logical_region(ctx, params_is, config.field_space);
    decoders[i].region =
      runtime->create_logical_region(ctx, params_is, config.field_space);
    for (int j = 0; j < config.numNodes * config.workersPerNode; j++) {
      encoders[i].gradients[j] =
        runtime->create_logical_region(ctx, params_is, config.field_space);
      decoders[i].gradients[j] =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    }
  }
  SharedVariable linear;
  {
    int numParams = (hidden_size + 1) * vocab_size;
    Rect<1> params_rect(Point<1>(0), Point<1>(numParams-1));
    IndexSpaceT<1> params_is = runtime->create_index_space(ctx, params_rect);
    linear.region = runtime->create_logical_region(ctx, params_is, config.field_space);
    // Create subregions for the shared variable linear
    for (int parts = 1; parts <= MAX_NUM_PARTS; parts *= 2) {
      Rect<1> rect(Point<1>(0), Point<1>(parts-1));
      IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
      IndexPartition ip = runtime->create_equal_partition(ctx, params_is, is);
      LogicalPartition lp = runtime->get_logical_partition(ctx, linear.region, ip);
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++, idx++) {
        DomainPoint dp(*it);
        linear.subregions[parts+idx]
          = runtime->get_logical_subregion_by_color(ctx, lp, dp);
      }
    }
    // Compute bboxes for the shared variable linear
    std::map<int, Rect<1> > bboxes;
    for (int i = 0; i < nodes_per_layer; i++) {
      ParallelConfig pc = global.linear[i];
      assert(pc.nDims == 2);
      for (int j = 0; j < pc.dim[1]; j++)
        for (int k = 0; k < pc.dim[0]; k++) {
          int gpuIdx = pc.gpu[j * pc.dim[0] +k];
          Rect<1> rect = runtime->get_index_space_domain(ctx,
                             linear.subregions[pc.dim[0]+k].get_index_space());
          if (bboxes.find(gpuIdx) == bboxes.end())
            bboxes[gpuIdx] = rect;
          else
            bboxes[gpuIdx] = bboxes[gpuIdx].union_bbox(rect);
        }
    }
    // TODO: Manually set bbox for the first GPU on each node
    for (int i = 0; i < config.numNodes * config.workersPerNode; i++) 
      if (bboxes.find(i) != bboxes.end()) {
        IndexSpaceT<1> params_is = runtime->create_index_space(ctx, bboxes[i]);
        linear.gradients[i] =
          runtime->create_logical_region(ctx, params_is, config.field_space);
      } else
        linear.gradients[i] = LogicalRegion::NO_REGION;
  }
  for (int i = 0; i < numLayers; i++) {
    // Add encoder lstm nodes
    for (int j = 0; j < nodes_per_layer; j++) {
      Tensor x = (i==0) ? srcs[j] : lstm[i-1][j].x;
      Tensor hx = (j==0) ? zero[i].hx : lstm[i][j-1].hx;
      Tensor cx = (j==0) ? zero[i].cx : lstm[i][j-1].cx;
      lstm[i][j] = add_lstm_node(x, hx, cx, global.lstm[i][j], encoders[i]);
    }
    // Add decoder lstm nodes
    for (int j = nodes_per_layer; j < 2*nodes_per_layer; j++) {
      Tensor x = (i==0) ? dsts[j-nodes_per_layer] : lstm[i-1][j].x;
      Tensor hx = lstm[i][j-1].hx;
      Tensor cx = lstm[i][j-1].cx;
      lstm[i][j] = add_lstm_node(x, hx, cx, global.lstm[i][j], decoders[i]);
    }
  }
  // Add linear nodes
  for (int j = nodes_per_layer; j < 2*nodes_per_layer; j++) {
    add_linear_node(lstm[numLayers-1][j].x, vocab_size,
                    global.linear[j-nodes_per_layer], linear);
  }
  // Add shared variables
  for (int i = 0; i < config.numLayers; i++) {
    sharedVariables.push_back(encoders[i]);
    sharedVariables.push_back(decoders[i]);
  }
  sharedVariables.push_back(linear);
}

void RnnModel::init()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->init(*this);
}

void RnnModel::forward()
{
  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->forward(*this);
  }
}

void RnnModel::backward()
{
  for (int i = layers.size() - 1; i >=0; i--) {
    layers[i]->backward(*this); 
  }
}

void RnnModel::update()
{
  for (int i = 0; i < sharedVariables.size(); i++)
    update_shared_variable(sharedVariables[i]);
}

/*
  regions[0]: (I/O): w
  regions[1..]: (O): w_grad
 */
void RnnModel::params_update_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == task->regions.size());
  float rate = *((float*)task->args);
  const AccessorRW<float, 1> acc_w(regions[0], FID_DATA);
  Rect<1> rect_w =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  for (int i = 1; i < regions.size(); i++) {
    const AccessorRO<float, 1> acc_w_grad(regions[i], FID_DATA);
    Rect<1> rect_w_grad =
      runtime->get_index_space_domain(ctx, task->regions[i].region.get_index_space());
    assert(rect_w.contains(rect_w_grad));
    assert(acc_w_grad.accessor.is_dense_arbitrary(rect_w_grad));
    float *w_ptr = acc_w.ptr(rect_w_grad.lo);
    const float *w_grad_ptr = acc_w_grad.ptr(rect_w_grad.lo);
    apply_add_with_scale<<<GET_BLOCKS(rect_w_grad.volume()), CUDA_NUM_THREADS>>>(
        w_ptr, w_grad_ptr, rect_w_grad.volume(), rate);
  }
}

void RnnModel::update_shared_variable(SharedVariable params)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  float rate = 1.0f;
  for (int node = 0; node < config.numNodes; node++) {
    TaskLauncher launcher(PARAMS_UPD_TASK_ID, TaskArgument(&rate, sizeof(rate)),
                          Predicate::TRUE_PRED, 0/*MapperID*/,
                          RnnMapper::assign_to_gpu(node * config.workersPerNode));
    int cnt = 0;
    for (int idx = 0; idx < config.workersPerNode; idx++) {
      int gpuIdx = node * config.workersPerNode + idx;
      LogicalRegion grad = params.gradients[gpuIdx];
      if (grad == LogicalRegion::NO_REGION) continue;
      launcher.add_region_requirement(
        RegionRequirement(grad, idx == 0 ? READ_WRITE:READ_ONLY, EXCLUSIVE, grad));
      launcher.add_field(cnt++, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }

  rate = 0.001f;
  TaskLauncher launcher(PARAMS_UPD_TASK_ID, TaskArgument(&rate, sizeof(rate)),
                        Predicate::TRUE_PRED, 0/*MapperID*/);
  launcher.add_region_requirement(
    RegionRequirement(params.region, READ_WRITE, EXCLUSIVE, params.region));
  launcher.add_field(0, FID_DATA);
  for (int node = 0; node < config.numNodes; node++) {
    int gpuIdx = node * config.workersPerNode;
    LogicalRegion grad = params.gradients[gpuIdx];
    int cnt = 1;
    if (grad == LogicalRegion::NO_REGION) continue;
    launcher.add_region_requirement(
      RegionRequirement(grad, READ_ONLY, EXCLUSIVE, grad));
    launcher.add_field(cnt++, FID_DATA);
  }
  runtime->execute_task(ctx, launcher);
}

