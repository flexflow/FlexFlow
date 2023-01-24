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

#include "../cnn_helper.h"
#include "rnn.h"
#include "rnn_mapper.h"

DnnHandle init_cudnn(Task const *task,
                     std::vector<PhysicalRegion> const &regions,
                     Context ctx,
                     HighLevelRuntime *runtime) {
  assert(regions.size() == 0);
  assert(task->arglen == sizeof(size_t));
  size_t workSpaceSize = *(size_t const *)task->args;
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

const SharedVariable SharedVariable::NO_VARIABLE = SharedVariable();

RnnOp::RnnOp(Tensor input, ParallelConfig pc, SharedVariable _params)
    : paraConfig(pc), params(_params) {
  inputs[0] = input;
}

RnnOp::RnnOp(
    Tensor t1, Tensor t2, Tensor t3, ParallelConfig pc, SharedVariable _params)
    : paraConfig(pc), params(_params) {
  inputs[0] = t1;
  inputs[1] = t2;
  inputs[2] = t3;
}

RnnOp::RnnOp(int n, Tensor *_inputs) {
  for (int i = 0; i < n; i++) {
    inputs[i] = _inputs[i];
  }
}

RnnModel::RnnModel(int batch_size,
                   int numLayers,
                   int seqLength,
                   int hidden_size,
                   int embed_size,
                   int vocab_size,
                   int num_parts,
                   int num_nodes,
                   int num_gpus_per_node,
                   GlobalConfig global,
                   Context ctx,
                   Runtime *runtime) {
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.batchSize = batch_size;
  config.hiddenSize = hidden_size;
  config.embedSize = embed_size;
  config.vocabSize = vocab_size;
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
  Rect<1> part_rect(Point<1>(0), Point<1>(num_parts - 1));
  part_is = runtime->create_index_space(ctx, part_rect);
  assert(seqLength <= MAX_SEQ_LENGTH);
  assert(numLayers <= MAX_NUM_LAYERS);
  int nodes_per_layer = seqLength / LSTM_PER_NODE_LENGTH;
  // Create srcs/dsts tensors
  {
    Rect<2> word_rect(Point<2>(0, 0),
                      Point<2>(batch_size - 1, LSTM_PER_NODE_LENGTH - 1));
    IndexSpaceT<2> word_is = runtime->create_index_space(ctx, word_rect);
    int extent_n = batch_size / num_parts;
    Rect<2, coord_t> extent(Point<2>(0, 0),
                            Point<2>(extent_n - 1, LSTM_PER_NODE_LENGTH - 1));
    Transform<2, 1, coord_t> trans;
    trans[0][0] = extent_n;
    trans[1][0] = 0;
    IndexPartition word_ip = runtime->create_partition_by_restriction(
        ctx, word_is, part_is, trans, extent);
    assert(runtime->is_index_partition_disjoint(ctx, word_ip));
    assert(runtime->is_index_partition_complete(ctx, word_ip));
    assert(seqLength % LSTM_PER_NODE_LENGTH == 0);
    for (int i = 0; i < nodes_per_layer; i++) {
      srcs[i].numDim = 2;
      srcs[i].adim[0] = batch_size;
      srcs[i].adim[1] = LSTM_PER_NODE_LENGTH;
      srcs[i].pdim[0] = extent_n;
      srcs[i].pdim[1] = LSTM_PER_NODE_LENGTH;
      srcs[i].region =
          runtime->create_logical_region(ctx, word_is, config.field_space);
      srcs[i].partition =
          runtime->get_logical_partition(ctx, srcs[i].region, word_ip);
      srcs[i].region_grad =
          runtime->create_logical_region(ctx, word_is, config.field_space);
      srcs[i].partition_grad =
          runtime->get_logical_partition(ctx, srcs[i].region_grad, word_ip);
      dsts[i] = srcs[i];
      dsts[i].region =
          runtime->create_logical_region(ctx, word_is, config.field_space);
      dsts[i].partition =
          runtime->get_logical_partition(ctx, dsts[i].region, word_ip);
      dsts[i].region_grad =
          runtime->create_logical_region(ctx, word_is, config.field_space);
      dsts[i].partition_grad =
          runtime->get_logical_partition(ctx, dsts[i].region_grad, word_ip);
    }
  }
  // Create zeroed tensors
  {
    Rect<2> hx_rect(Point<2>(0, 0), Point<2>(hidden_size - 1, batch_size - 1));
    IndexSpaceT<2> hx_is = runtime->create_index_space(ctx, hx_rect);
    int extent_c = hidden_size;
    int extent_n = batch_size / num_parts;
    Rect<2> hx_ext(Point<2>(0, 0), Point<2>(extent_c - 1, extent_n - 1));
    Transform<2, 1, coord_t> hx_trans;
    hx_trans[0][0] = 0;
    hx_trans[1][0] = extent_n;
    IndexPartition hx_ip = runtime->create_partition_by_restriction(
        ctx, hx_is, part_is, hx_trans, hx_ext);
    assert(runtime->is_index_partition_disjoint(ctx, hx_ip));
    assert(runtime->is_index_partition_complete(ctx, hx_ip));
    for (int i = 0; i < numLayers; i++) {
      for (int j = 0; j < 2; j++) {
        Tensor t;
        t.numDim = 2;
        t.adim[0] = hidden_size;
        t.adim[1] = batch_size;
        t.pdim[0] = extent_c;
        t.pdim[1] = extent_n;
        t.region =
            runtime->create_logical_region(ctx, hx_is, config.field_space);
        t.partition = runtime->get_logical_partition(ctx, t.region, hx_ip);
        t.region_grad =
            runtime->create_logical_region(ctx, hx_is, config.field_space);
        t.partition_grad =
            runtime->get_logical_partition(ctx, t.region_grad, hx_ip);
        if (j == 0) {
          zero[i].hx = t;
        } else {
          zero[i].cx = t;
        }
      }
    }
  }
  // Embedding
  SharedVariable srcEmbed, dstEmbed;
  {
    int numParams = config.vocabSize * config.embedSize;
    Rect<1> params_rect(Point<1>(0), Point<1>(numParams - 1));
    IndexSpaceT<1> params_is = runtime->create_index_space(ctx, params_rect);
    srcEmbed.region =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    dstEmbed.region =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    for (int i = 0; i < 2 * nodes_per_layer; i++) {
      ParallelConfig pc = global.embed[i];
      assert(pc.nDims == 1);
      for (int j = 0; j < pc.dim[0]; j++) {
        int gpuId = pc.gpu[j];
        if (i < nodes_per_layer) {
          if (srcEmbed.gradients[gpuId] == LogicalRegion::NO_REGION) {
            srcEmbed.gradients[gpuId] = runtime->create_logical_region(
                ctx, params_is, config.field_space);
          }
        } else {
          if (dstEmbed.gradients[gpuId] == LogicalRegion::NO_REGION) {
            dstEmbed.gradients[gpuId] = runtime->create_logical_region(
                ctx, params_is, config.field_space);
          }
        }
      }
    }
    // Collect masterOnNode for srcEmbed/dstEmbed
    for (int i = 0; i < config.numNodes; i++) {
      for (int j = config.workersPerNode - 1; j >= 0; j--) {
        int gpuId = i * config.workersPerNode + j;
        if (srcEmbed.gradients[gpuId] != LogicalRegion::NO_REGION) {
          srcEmbed.masterOnNode[i] = gpuId;
        }
        if (dstEmbed.gradients[gpuId] != LogicalRegion::NO_REGION) {
          dstEmbed.masterOnNode[i] = gpuId;
        }
      }
    }
  }

  // Encoders/decoders
  SharedVariable encoders[MAX_NUM_LAYERS], decoders[MAX_NUM_LAYERS];
  for (int i = 0; i < numLayers; i++) {
    int input_size = (i == 0) ? embed_size : hidden_size;
    int output_size = hidden_size;
    int numParams = (input_size + 1 + output_size + 1) * output_size * 4;
    Rect<1> params_rect(Point<1>(0), Point<1>(numParams - 1));
    IndexSpaceT<1> params_is = runtime->create_index_space(ctx, params_rect);
    encoders[i].region =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    decoders[i].region =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    for (int j = 0; j < 2 * nodes_per_layer; j++) {
      ParallelConfig pc = global.lstm[i][j];
      assert(pc.nDims == 1);
      for (int k = 0; k < pc.dim[0]; k++) {
        int gpuId = pc.gpu[k];
        if (j < nodes_per_layer) {
          if (encoders[i].gradients[gpuId] == LogicalRegion::NO_REGION) {
            encoders[i].gradients[gpuId] = runtime->create_logical_region(
                ctx, params_is, config.field_space);
          }
        } else {
          if (decoders[i].gradients[gpuId] == LogicalRegion::NO_REGION) {
            decoders[i].gradients[gpuId] = runtime->create_logical_region(
                ctx, params_is, config.field_space);
          }
        }
      }
    }
    // Collect masterOnNode for encoders[i]/decoders[i]
    for (int j = 0; j < config.numNodes; j++) {
      for (int k = config.workersPerNode - 1; k >= 0; k--) {
        int gpuId = j * config.workersPerNode + k;
        if (encoders[i].gradients[gpuId] != LogicalRegion::NO_REGION) {
          encoders[i].masterOnNode[j] = gpuId;
        }
        if (decoders[i].gradients[gpuId] != LogicalRegion::NO_REGION) {
          decoders[i].masterOnNode[j] = gpuId;
        }
      }
    }
  }
  SharedVariable linear;
  {
    int numParams = (hidden_size + 1) * vocab_size;
    Rect<1> params_rect(Point<1>(0), Point<1>(numParams - 1));
    IndexSpaceT<1> params_is = runtime->create_index_space(ctx, params_rect);
    linear.region =
        runtime->create_logical_region(ctx, params_is, config.field_space);
    linear.subregions[1] = linear.region;
    // Create subregions for the shared variable linear
    for (int parts = 2; parts <= MAX_NUM_PARTS; parts *= 2) {
      Rect<1> rect(Point<1>(0), Point<1>(parts - 1));
      IndexSpaceT<1> is = runtime->create_index_space(ctx, rect);
      IndexPartition ip = runtime->create_equal_partition(ctx, params_is, is);
      LogicalPartition lp =
          runtime->get_logical_partition(ctx, linear.region, ip);
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++, idx++) {
        DomainPoint dp(*it);
        linear.subregions[parts + idx] =
            runtime->get_logical_subregion_by_color(ctx, lp, dp);
      }
    }
    // Compute bboxes for the shared variable linear
    // Also compute masterOnNode which is the largest gradients on each node
    std::map<int, Rect<1>> bboxes;
    for (int i = 0; i < nodes_per_layer; i++) {
      ParallelConfig pc = global.linear[i];
      assert(pc.nDims == 2);
      for (int j = 0; j < pc.dim[1]; j++) {
        for (int k = 0; k < pc.dim[0]; k++) {
          int gpuIdx = pc.gpu[j * pc.dim[0] + k];
          Rect<1> rect = runtime->get_index_space_domain(
              ctx, linear.subregions[pc.dim[0] + k].get_index_space());
          if (bboxes.find(gpuIdx) == bboxes.end()) {
            bboxes[gpuIdx] = rect;
          } else {
            bboxes[gpuIdx] = bboxes[gpuIdx].union_bbox(rect);
          }
          int nodeIdx = gpuIdx / config.workersPerNode;
          if (linear.masterOnNode[nodeIdx] == MASTER_NOT_ASSIGNED) {
            linear.masterOnNode[nodeIdx] = gpuIdx;
          } else {
            int masterIdx = linear.masterOnNode[nodeIdx];
            if (bboxes[gpuIdx].volume() > bboxes[masterIdx].volume()) {
              linear.masterOnNode[nodeIdx] = gpuIdx;
            }
          }
        }
      }
    }
    // The first bbox on each node is a superset of all bboxes on that node
    for (int n = 0; n < config.numNodes; n++) {
      if (linear.masterOnNode[n] != MASTER_NOT_ASSIGNED) {
        for (int j = 0; j < config.workersPerNode; j++) {
          if (bboxes.find(n * config.workersPerNode + j) != bboxes.end()) {
            Rect<1> rect = bboxes[n * config.workersPerNode + j];
            bboxes[linear.masterOnNode[n]] =
                bboxes[linear.masterOnNode[n]].union_bbox(rect);
          }
        }
      }
    }
    for (int i = 0; i < config.numNodes * config.workersPerNode; i++) {
      if (bboxes.find(i) != bboxes.end()) {
        IndexSpaceT<1> params_is = runtime->create_index_space(ctx, bboxes[i]);
        linear.gradients[i] =
            runtime->create_logical_region(ctx, params_is, config.field_space);
      } else {
        linear.gradients[i] = LogicalRegion::NO_REGION;
      }
    }
  }

  Tensor embed[2 * MAX_SEQ_LENGTH];
  for (int i = 0; i < 2 * nodes_per_layer; i++) {
    embed[i] = add_embed_node(i < nodes_per_layer ? srcs[i]
                                                  : dsts[i - nodes_per_layer],
                              config.vocabSize,
                              config.embedSize,
                              global.embed[i],
                              i < nodes_per_layer ? srcEmbed : dstEmbed);
  }
  for (int i = 0; i < numLayers; i++) {
    // Add encoder lstm nodes
    for (int j = 0; j < nodes_per_layer; j++) {
      Tensor x = (i == 0) ? embed[j] : lstm[i - 1][j].x;
      Tensor hx = (j == 0) ? zero[i].hx : lstm[i][j - 1].hx;
      Tensor cx = (j == 0) ? zero[i].cx : lstm[i][j - 1].cx;
      lstm[i][j] = add_lstm_node(x, hx, cx, global.lstm[i][j], encoders[i]);
    }
    // Add decoder lstm nodes
    for (int j = nodes_per_layer; j < 2 * nodes_per_layer; j++) {
      Tensor x = (i == 0) ? embed[j] : lstm[i - 1][j].x;
      Tensor hx = lstm[i][j - 1].hx;
      Tensor cx = lstm[i][j - 1].cx;
      lstm[i][j] = add_lstm_node(x, hx, cx, global.lstm[i][j], decoders[i]);
    }
  }
  // Add linear nodes
  for (int j = nodes_per_layer; j < 2 * nodes_per_layer; j++) {
    Tensor logit = add_linear_node(lstm[numLayers - 1][j].x,
                                   vocab_size,
                                   global.linear[j - nodes_per_layer],
                                   linear);
    add_softmaxDP_node(
        logit, dsts[j - nodes_per_layer], global.softmax[j - nodes_per_layer]);
  }

  // Add shared variables
  sharedVariables.push_back(srcEmbed);
  sharedVariables.push_back(dstEmbed);
  for (int i = 0; i < config.numLayers; i++) {
    sharedVariables.push_back(encoders[i]);
    sharedVariables.push_back(decoders[i]);
  }
  sharedVariables.push_back(linear);
}

void RnnModel::word_init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  Rect<2> rect0 = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  int *host_ptr;
  bool same = *((bool *)task->args);
  checkCUDA(cudaHostAlloc(&host_ptr,
                          sizeof(int) * rect0.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  for (int i = 0; i < rect0.volume(); i++) {
    host_ptr[i] = same ? 1 : i % 16;
  }
  for (int i = 0; i < regions.size(); i++) {
    AccessorWO<int, 2> const acc(regions[i], FID_DATA);
    Rect<2> rect = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    assert(acc.accessor.is_dense_arbitrary(rect));
    assert(rect == rect0);
    int *ptr = acc.ptr(rect.lo);
    checkCUDA(cudaMemcpy(
        ptr, host_ptr, sizeof(int) * rect0.volume(), cudaMemcpyHostToDevice));
  }
  checkCUDA(cudaFreeHost(host_ptr));
}

void RnnModel::init() {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Init words
  Rect<1> part_rect = runtime->get_index_space_domain(ctx, part_is);
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    int idx = 0;
    bool same = false;
    TaskLauncher launcher(WORD_INIT_TASK_ID,
                          TaskArgument(&same, sizeof(same)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(0));
    DomainPoint dp(*it);
    for (int i = 0; i * LSTM_PER_NODE_LENGTH < config.seqLength; i++) {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(srcs[i].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, WRITE_ONLY, EXCLUSIVE, srcs[i].region));
      launcher.add_field(idx++, FID_DATA);
    }
    for (int i = 0; i * LSTM_PER_NODE_LENGTH < config.seqLength; i++) {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(dsts[i].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, WRITE_ONLY, EXCLUSIVE, dsts[i].region));
      launcher.add_field(idx++, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  // Init zero tensors
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    int idx = 0;
    TaskLauncher launcher(ZERO_2D_INIT_TASK_ID,
                          TaskArgument(NULL, 0),
                          Predicate::TRUE_PRED,
                          0,
                          RnnMapper::assign_to_gpu(0));
    DomainPoint dp(*it);
    for (int i = 0; i < config.numLayers; i++) {
      LogicalRegion hx =
          runtime->get_logical_subregion_by_color(zero[i].hx.partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(hx, WRITE_ONLY, EXCLUSIVE, zero[i].hx.region));
      launcher.add_field(idx++, FID_DATA);
    }
    for (int i = 0; i < config.numLayers; i++) {
      LogicalRegion cx =
          runtime->get_logical_subregion_by_color(zero[i].cx.partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(cx, WRITE_ONLY, EXCLUSIVE, zero[i].cx.region));
      launcher.add_field(idx++, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  // Init hx_grad/cx_grad for the last LSTM node on each layer
  int nodes_per_layer = config.seqLength / LSTM_PER_NODE_LENGTH;
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    int idx = 0;
    TaskLauncher launcher(ZERO_2D_INIT_TASK_ID,
                          TaskArgument(NULL, 0),
                          Predicate::TRUE_PRED,
                          0,
                          RnnMapper::assign_to_gpu(0));
    DomainPoint dp(*it);
    for (int i = 0; i < config.numLayers; i++) {
      LSTMTensors last_lstm = lstm[i][2 * nodes_per_layer - 1];
      // hx
      LogicalRegion hx_grad = runtime->get_logical_subregion_by_color(
          last_lstm.hx.partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(
          hx_grad, WRITE_ONLY, EXCLUSIVE, last_lstm.hx.region_grad));
      launcher.add_field(idx++, FID_DATA);
      // cx
      LogicalRegion cx_grad = runtime->get_logical_subregion_by_color(
          last_lstm.cx.partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(
          cx_grad, WRITE_ONLY, EXCLUSIVE, last_lstm.cx.region_grad));
      launcher.add_field(idx++, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  // TODO: to be removed when we have attention layers
  // Init y_grad for the decoder lstm nodes
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    int idx = 0;
    TaskLauncher launcher(ZERO_3D_INIT_TASK_ID,
                          TaskArgument(NULL, 0),
                          Predicate::TRUE_PRED,
                          0,
                          RnnMapper::assign_to_gpu(0));
    DomainPoint dp(*it);
    for (int i = 0; i < nodes_per_layer; i++) {
      LSTMTensors top_lstm = lstm[config.numLayers - 1][i];
      LogicalRegion y_grad = runtime->get_logical_subregion_by_color(
          top_lstm.x.partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(
          y_grad, WRITE_ONLY, EXCLUSIVE, top_lstm.x.region_grad));
      launcher.add_field(idx++, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  // Init shared variables
  for (int i = 0; i < sharedVariables.size(); i++) {
    init_shared_variable(sharedVariables[i]);
  }
  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->init(*this);
  }
}

void RnnModel::zero_3d_init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  for (int i = 0; i < task->regions.size(); i++) {
    AccessorWO<float, 3> const acc_w(regions[i], FID_DATA);
    Rect<3> rect_w = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    assert(acc_w.accessor.is_dense_arbitrary(rect_w));
    float *w_ptr = acc_w.ptr(rect_w.lo);
    assign_kernel<<<GET_BLOCKS(rect_w.volume()), CUDA_NUM_THREADS>>>(
        w_ptr, rect_w.volume(), 0.0f);
  }
}

void RnnModel::zero_2d_init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  for (int i = 0; i < task->regions.size(); i++) {
    AccessorWO<float, 2> const acc_w(regions[i], FID_DATA);
    Rect<2> rect_w = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    assert(acc_w.accessor.is_dense_arbitrary(rect_w));
    float *w_ptr = acc_w.ptr(rect_w.lo);
    assign_kernel<<<GET_BLOCKS(rect_w.volume()), CUDA_NUM_THREADS>>>(
        w_ptr, rect_w.volume(), 0.0f);
  }
}

void RnnModel::zero_1d_init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  for (int i = 0; i < task->regions.size(); i++) {
    AccessorWO<float, 1> const acc_w(regions[i], FID_DATA);
    Rect<1> rect_w = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    assert(acc_w.accessor.is_dense_arbitrary(rect_w));
    float *w_ptr = acc_w.ptr(rect_w.lo);
    assign_kernel<<<GET_BLOCKS(rect_w.volume()), CUDA_NUM_THREADS>>>(
        w_ptr, rect_w.volume(), 0.0f);
  }
}

void RnnModel::dummy_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {}

void RnnModel::forward() {
  config.iterator++;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Step 1: launch dummy tasks to prefetch shared variables
  for (size_t i = 0; i < sharedVariables.size(); i++) {
    for (int n = 0; n < config.numNodes; n++) {
      if (sharedVariables[i].masterOnNode[n] != MASTER_NOT_ASSIGNED) {
        int gpuId = sharedVariables[i].masterOnNode[n];
        TaskLauncher launcher(DUMMY_TASK_ID,
                              TaskArgument(NULL, 0),
                              Predicate::TRUE_PRED,
                              0,
                              RnnMapper::assign_to_gpu(gpuId));
        launcher.add_region_requirement(
            RegionRequirement(sharedVariables[i].region,
                              READ_ONLY,
                              EXCLUSIVE,
                              sharedVariables[i].region));
        launcher.add_field(0, FID_DATA);
        runtime->execute_task(ctx, launcher);
      }
    }
  }
  runtime->issue_mapping_fence(ctx);
  // Step 2: zero gradients
  for (size_t i = 0; i < sharedVariables.size(); i++) {
    for (int j = 0; j < config.workersPerNode * config.numNodes; j++) {
      if (sharedVariables[i].gradients[j] != LogicalRegion::NO_REGION) {
        TaskLauncher launcher(ZERO_1D_INIT_TASK_ID,
                              TaskArgument(NULL, 0),
                              Predicate::TRUE_PRED,
                              0,
                              RnnMapper::assign_to_gpu(j));
        LogicalRegion gradient = sharedVariables[i].gradients[j];
        launcher.add_region_requirement(
            RegionRequirement(gradient, WRITE_ONLY, EXCLUSIVE, gradient));
        launcher.add_field(0, FID_DATA);
        runtime->execute_task(ctx, launcher);
      }
    }
  }
  // Step 3: launch forward tasks
  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->forward(*this);
  }
}

void RnnModel::backward() {
  for (int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backward(*this);
  }
}

void RnnModel::update() {
  for (int i = sharedVariables.size() - 1; i >= 0; i--) {
    update_shared_variable(sharedVariables[i]);
  }
}

/*
  regions[0](O): w
*/
void RnnModel::params_init_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  float value = *((float *)task->args);
  AccessorWO<float, 1> const acc_w(regions[0], FID_DATA);
  Rect<1> rect_w = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  float *w_ptr = acc_w.ptr(rect_w.lo);
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  curandGenerator_t genGPU;
  curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(genGPU, stream);
  curandSetPseudoRandomGeneratorSeed(genGPU, 1234LL);
  curandGenerateUniform(genGPU, w_ptr, rect_w.volume());
  checkCUDA(cudaDeviceSynchronize());
  scale_kernel<<<GET_BLOCKS(rect_w.volume()), CUDA_NUM_THREADS>>>(
      w_ptr, rect_w.volume(), -value, value);
  // assign_kernel<<<GET_BLOCKS(rect_w.volume()), CUDA_NUM_THREADS>>>(
  //   w_ptr, rect_w.volume(), value);
}

void RnnModel::init_shared_variable(SharedVariable params) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  float value = 0.1f;
  TaskLauncher launcher(PARAMS_INIT_TASK_ID,
                        TaskArgument(&value, sizeof(value)),
                        Predicate::TRUE_PRED,
                        0 /*MapperID*/,
                        RnnMapper::assign_to_gpu(params.masterOnNode[0]));
  launcher.add_region_requirement(
      RegionRequirement(params.region, WRITE_ONLY, EXCLUSIVE, params.region));
  launcher.add_field(0, FID_DATA);
  Future f = runtime->execute_task(ctx, launcher);
  f.get_void_result();
}

/*
  regions[0]: (I/O): w
  regions[1..]: (O): w_grad
 */
void RnnModel::params_update_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == task->regions.size());
  float rate = *((float *)task->args);
  AccessorRW<float, 1> const acc_w(regions[0], FID_DATA);
  Rect<1> rect_w = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  for (int i = 1; i < regions.size(); i++) {
    AccessorRO<float, 1> const acc_w_grad(regions[i], FID_DATA);
    Rect<1> rect_w_grad = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    assert(rect_w.contains(rect_w_grad));
    assert(acc_w_grad.accessor.is_dense_arbitrary(rect_w_grad));
    float *w_ptr = acc_w.ptr(rect_w_grad.lo);
    float const *w_grad_ptr = acc_w_grad.ptr(rect_w_grad.lo);
    apply_add_with_scale<<<GET_BLOCKS(rect_w_grad.volume()),
                           CUDA_NUM_THREADS>>>(
        w_ptr, w_grad_ptr, rect_w_grad.volume(), rate);
#ifdef PRINT_INTERMEDIATE_RESULT
    print_tensor<1, float>(w_grad_ptr, rect_w_grad, "partial_w");
#endif
  }
#ifdef PRINT_INTERMEDIATE_RESULT
  float *w_ptr = acc_w.ptr(rect_w.lo);
  print_tensor<1, float>(w_ptr, rect_w, "final_w");
#endif
}

void RnnModel::update_shared_variable(SharedVariable params) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // for (int i = 0; i < config.workersPerNode; i++)
  //   if (params.gradients[i] != LogicalRegion::NO_REGION) {
  //     Rect<1> rect =
  //       runtime->get_index_space_domain(ctx,
  //       params.gradients[i].get_index_space());
  //     printf("rect[%d]: lo(%d) hi(%d)\n", i, rect.lo[0], rect.hi[0]);
  //   }
  float rate = 1.0f;
  for (int node = 0; node < config.numNodes; node++) {
    if (params.masterOnNode[node] != MASTER_NOT_ASSIGNED) {
      TaskLauncher launcher(
          PARAMS_UPD_TASK_ID,
          TaskArgument(&rate, sizeof(rate)),
          Predicate::TRUE_PRED,
          0 /*MapperID*/,
          RnnMapper::assign_to_gpu(params.masterOnNode[node]));
      LogicalRegion masterGrad = params.gradients[params.masterOnNode[node]];
      assert(masterGrad != LogicalRegion::NO_REGION);
      launcher.add_region_requirement(
          RegionRequirement(masterGrad, READ_WRITE, EXCLUSIVE, masterGrad));
      launcher.add_field(0, FID_DATA);
      int cnt = 1;
      for (int idx = 0; idx < config.workersPerNode; idx++) {
        int gpuIdx = node * config.workersPerNode + idx;
        if (gpuIdx == params.masterOnNode[node]) {
          continue;
        }
        LogicalRegion grad = params.gradients[gpuIdx];
        if (grad == LogicalRegion::NO_REGION) {
          continue;
        }
        launcher.add_region_requirement(
            RegionRequirement(grad, READ_ONLY, EXCLUSIVE, grad));
        launcher.add_field(cnt++, FID_DATA);
      }
      // printf("Step 1: cnt = %d\n", cnt);
      runtime->execute_task(ctx, launcher);
    }
  }
  rate = -0.1f;
  TaskLauncher launcher(PARAMS_UPD_TASK_ID,
                        TaskArgument(&rate, sizeof(rate)),
                        Predicate::TRUE_PRED,
                        0 /*MapperID*/,
                        RnnMapper::assign_to_gpu(params.masterOnNode[0]));
  launcher.add_region_requirement(
      RegionRequirement(params.region, READ_WRITE, EXCLUSIVE, params.region));
  launcher.add_field(0, FID_DATA);
  int cnt = 1;
  for (int node = 0; node < config.numNodes; node++) {
    if (params.masterOnNode[node] != MASTER_NOT_ASSIGNED) {
      int gpuIdx = params.masterOnNode[node];
      LogicalRegion grad = params.gradients[gpuIdx];
      assert(grad != LogicalRegion::NO_REGION);
      launcher.add_region_requirement(
          RegionRequirement(grad, READ_ONLY, EXCLUSIVE, grad));
      launcher.add_field(cnt++, FID_DATA);
    }
  }
  // printf("Step 2: cnt = %d\n", cnt);
  runtime->execute_task(ctx, launcher);
}
