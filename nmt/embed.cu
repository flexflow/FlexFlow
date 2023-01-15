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

struct EmbedInitParams {
  DnnHandle handle;
  int batchSize, outputSize, vocabSize;
};

Tensor RnnModel::add_embed_node(Tensor x,
                                int vocab_size,
                                int output_size,
                                ParallelConfig pc,
                                SharedVariable params) {
  assert(x.numDim == 2);
  assert(x.adim[1] == LSTM_PER_NODE_LENGTH);
  assert(x.pdim[1] == LSTM_PER_NODE_LENGTH);
  Embed *node = new Embed(config, x, vocab_size, output_size, pc, params);
  layers.push_back(node);
  return node->outputs[0];
}

Embed::Embed(RnnConfig config,
             Tensor x,
             int _vocab_size,
             int _output_size,
             ParallelConfig pc,
             SharedVariable _params)
    : RnnOp(x, pc, _params), batchSize(x.adim[0]), vocabSize(_vocab_size),
      outputSize(_output_size) {
  Context ctx = config.lg_ctx;
  HighLevelRuntime *runtime = config.lg_hlr;
  assert(pc.nDims == 1);
  {
    Rect<1> rect(Point<1>(0), Point<1>(pc.dim[0] - 1));
    part_rect = rect;
  }
  IndexSpaceT<1> part_is = runtime->create_index_space(ctx, part_rect);
  FieldSpace fs = config.field_space;
  Rect<3, coord_t> y_rect(
      Point<3>(0, 0, 0),
      Point<3>(outputSize - 1, batchSize - 1, LSTM_PER_NODE_LENGTH - 1));
  IndexSpaceT<3> y_is = runtime->create_index_space(ctx, y_rect);
  LogicalRegion y_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion y_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  int num_par_n = part_rect.hi[0] - part_rect.lo[0] + 1;
  assert(batchSize % num_par_n == 0);
  int extent_n = batchSize / num_par_n;
  int extent_c = outputSize;
  Rect<3, coord_t> extent(
      Point<3>(0, 0, 0),
      Point<3>(extent_c - 1, extent_n - 1, LSTM_PER_NODE_LENGTH - 1));
  Transform<3, 1, coord_t> trans;
  trans[0][0] = 0;
  trans[1][0] = extent_n;
  trans[2][0] = 0;
  IndexPartition y_ip = runtime->create_partition_by_restriction(
      ctx, y_is, part_is, trans, extent);
  assert(runtime->is_index_partition_disjoint(ctx, y_ip));
  assert(runtime->is_index_partition_complete(ctx, y_ip));
  LogicalPartition y_lp = runtime->get_logical_partition(ctx, y_lr, y_ip);
  LogicalPartition y_grad_lp =
      runtime->get_logical_partition(ctx, y_grad_lr, y_ip);
  outputs[0].region = y_lr;
  outputs[0].region_grad = y_grad_lr;
  outputs[0].partition = y_lp;
  outputs[0].partition_grad = y_grad_lp;
  outputs[0].numDim = 3;
  outputs[0].adim[0] = outputSize;
  outputs[0].adim[1] = batchSize;
  outputs[0].adim[2] = LSTM_PER_NODE_LENGTH;
  outputs[0].pdim[0] = extent_c;
  outputs[0].pdim[1] = extent_n;
  outputs[0].pdim[2] = LSTM_PER_NODE_LENGTH;
}

/*
  regions[0] (I): x
  regions[1] (I): w
  regions[2] (O): y
 */
OpMeta *Embed::init_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  EmbedInitParams const *embed = (EmbedInitParams *)task->args;
  Rect<2> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(rect_x.hi[0] - rect_x.lo[0] + 1 == embed->batchSize);
  assert(rect_x.hi[1] - rect_x.lo[1] + 1 == LSTM_PER_NODE_LENGTH);
  assert(rect_w.hi[0] - rect_w.lo[0] + 1 ==
         embed->vocabSize * embed->outputSize);
  assert(rect_y.hi[0] - rect_y.lo[0] + 1 == embed->outputSize);
  assert(rect_y.hi[1] - rect_y.lo[1] + 1 == embed->batchSize);
  assert(rect_y.hi[2] - rect_y.lo[2] + 1 == LSTM_PER_NODE_LENGTH);
  EmbedMeta *m = new EmbedMeta(embed->handle);
  m->profiling_runtime = false;
  return m;
}

void Embed::init(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    EmbedInitParams initParams;
    initParams.handle = model.dnn_handlers[paraConfig.gpu[idx]];
    initParams.batchSize = outputs[0].pdim[1];
    initParams.outputSize = outputs[0].pdim[0];
    initParams.vocabSize = vocabSize;
    // batch is the first dim of input and the second dim of output
    assert(inputs[0].pdim[0] == outputs[0].pdim[1]);
    TaskLauncher launcher(EMBED_INIT_TASK_ID,
                          TaskArgument(&initParams, sizeof(initParams)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(inputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.region, READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(1, FID_DATA);
    {
      LogicalRegion y =
          runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, WRITE_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(2, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    meta[idx] = f.get_result<OpMeta *>();
  }
}

__global__ void embedForward(int const *x_ptr,
                             float const *embed,
                             float *y_ptr,
                             coord_t numElements,
                             int shift,
                             int outputSize) {
  CUDA_KERNEL_LOOP(i, numElements) {
    int idx = i >> shift;
    int off = i & (outputSize - 1);
    int wordIdx = x_ptr[idx];
    y_ptr[i] = embed[(wordIdx << shift) + off];
  }
}

__global__ void embedBackward(int const *x_ptr,
                              float *embed,
                              float const *y_ptr,
                              coord_t numElements,
                              int shift,
                              int outputSize) {
  CUDA_KERNEL_LOOP(i, numElements) {
    int idx = i >> shift;
    int off = i & (outputSize - 1);
    int wordIdx = x_ptr[idx];
    atomicAdd(embed + (wordIdx << shift) + off, y_ptr[i]);
  }
}

/*
  regions[0](I): x
  regions[1](I): w
  regions[2](O): y
*/
void Embed::forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  EmbedMeta const *m = *((EmbedMeta **)task->args);
  AccessorRO<int, 2> const acc_x(regions[0], FID_DATA);
  AccessorRO<float, 1> const acc_w(regions[1], FID_DATA);
  AccessorWO<float, 3> const acc_y(regions[2], FID_DATA);
  Rect<2> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  int batch_size = rect_y.hi[1] - rect_y.lo[1] + 1;
  int output_size = rect_y.hi[0] - rect_y.lo[0] + 1;
  int const *x_ptr = acc_x.ptr(rect_x.lo);
  float const *w_ptr = acc_w.ptr(rect_w.lo);
  float *y_ptr = acc_y.ptr(rect_y.lo);
  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  int shift = 0;
  int size = 1;
  while (size < output_size) {
    size = size * 2;
    shift = shift + 1;
  }
  assert(size == output_size);
  embedForward<<<GET_BLOCKS(rect_y.volume()), CUDA_NUM_THREADS>>>(
      x_ptr, w_ptr, y_ptr, rect_y.volume(), shift, output_size);
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Embed forward time = %.2lfms\n", elapsed);
  }
#endif
}

void Embed::forward(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    OpMeta *mp = meta[idx];
    TaskLauncher launcher(EMBED_FWD_TASK_ID,
                          TaskArgument(&mp, sizeof(OpMeta *)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(inputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.region, READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(1, FID_DATA);
    {
      LogicalRegion y =
          runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, WRITE_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(2, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }
}

/*
  regions[0](I): x
  regions[1](I/O): w_grad
  regions[2](I): y_grad
*/
void Embed::backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  EmbedMeta const *m = *((EmbedMeta **)task->args);
  AccessorRO<int, 2> const acc_x(regions[0], FID_DATA);
  AccessorRW<float, 1> const acc_w(regions[1], FID_DATA);
  AccessorRO<float, 3> const acc_y(regions[2], FID_DATA);
  Rect<2> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  int batch_size = rect_y.hi[1] - rect_y.lo[1] + 1;
  int output_size = rect_y.hi[0] - rect_y.lo[0] + 1;
  int const *x_ptr = acc_x.ptr(rect_x.lo);
  float *w_ptr = acc_w.ptr(rect_w.lo);
  float const *y_ptr = acc_y.ptr(rect_y.lo);
  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  int shift = 0;
  int size = 1;
  while (size < output_size) {
    size = size * 2;
    shift = shift + 1;
  }
  assert(size == output_size);
  embedBackward<<<GET_BLOCKS(rect_y.volume()), CUDA_NUM_THREADS>>>(
      x_ptr, w_ptr, y_ptr, rect_y.volume(), shift, output_size);
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Embed backward time = %.2lfms\n", elapsed);
  }
#endif
}

void Embed::backward(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    OpMeta *mp = meta[idx];
    TaskLauncher launcher(EMBED_BWD_TASK_ID,
                          TaskArgument(&mp, sizeof(OpMeta *)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(inputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.gradients[paraConfig.gpu[idx]],
                          READ_WRITE,
                          EXCLUSIVE,
                          params.gradients[paraConfig.gpu[idx]]));
    launcher.add_field(1, FID_DATA);
    {
      LogicalRegion y_grad = runtime->get_logical_subregion_by_color(
          outputs[0].partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(
          y_grad, READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
      launcher.add_field(2, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }
}

void Embed::update(RnnModel const &model) {}
