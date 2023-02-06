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

struct SoftmaxDPInitParams {
  DnnHandle handle;
  int batchSize;
  bool profiling;
};

Tensor RnnModel::add_softmaxDP_node(Tensor logit,
                                    Tensor label,
                                    ParallelConfig pc) {
  assert(logit.numDim == 3);
  assert(logit.adim[2] == LSTM_PER_NODE_LENGTH);
  assert(logit.pdim[2] == LSTM_PER_NODE_LENGTH);
  SoftmaxDP *node = new SoftmaxDP(config, logit, label, pc);
  layers.push_back(node);
  return node->outputs[0];
}

SoftmaxDP::SoftmaxDP(RnnConfig config,
                     Tensor logit,
                     Tensor _label,
                     ParallelConfig pc)
    : RnnOp(logit, pc, SharedVariable::NO_VARIABLE), label(_label) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  assert(pc.nDims == 1);
  int num_par_n = pc.dim[0];
  {
    Rect<1> rect(Point<1>(0), Point<1>(num_par_n - 1));
    part_rect = rect;
  }
  IndexSpaceT<1> part_is = runtime->create_index_space(ctx, part_rect);
  int batch_size = logit.adim[1];
  int output_size = logit.adim[0];
  FieldSpace fs = config.field_space;
  Rect<3, coord_t> y_rect(
      Point<3>(0, 0, 0),
      Point<3>(output_size - 1, batch_size - 1, LSTM_PER_NODE_LENGTH - 1));
  IndexSpaceT<3> y_is = runtime->create_index_space(ctx, y_rect);
  LogicalRegion y_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion y_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  assert(batch_size % num_par_n == 0);
  int extent_n = batch_size / num_par_n;
  Rect<3, coord_t> extent(
      Point<3>(0, 0, 0),
      Point<3>(output_size - 1, extent_n - 1, LSTM_PER_NODE_LENGTH - 1));
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
  outputs[0].numDim = 3;
  outputs[0].adim[0] = output_size;
  outputs[0].adim[1] = batch_size;
  outputs[0].adim[2] = LSTM_PER_NODE_LENGTH;
  outputs[0].pdim[0] = output_size;
  outputs[0].pdim[1] = extent_n;
  outputs[0].pdim[2] = LSTM_PER_NODE_LENGTH;
  outputs[0].region = y_lr;
  outputs[0].partition = y_lp;
  outputs[0].region_grad = y_grad_lr;
  outputs[0].partition_grad = y_grad_lp;
  // Every partition reads all input_channels
  // Use the same partitioning as outputs
  // if (inputs[0].pdim[0] == outputs[0].pdim[0]
  //  && inputs[0].pdim[1] == outputs[0].pdim[1]) {
  //  logit_lp = inputs[0].partition;
  //  logit_grad_lp = inputs[0].partition_grad;
  //} else {
  IndexSpaceT<3> logit_is(inputs[0].region.get_index_space());
  IndexPartition logit_ip = runtime->create_partition_by_restriction(
      ctx, logit_is, part_is, trans, extent);
  logit_lp = runtime->get_logical_partition(ctx, inputs[0].region, logit_ip);
  logit_grad_lp =
      runtime->get_logical_partition(ctx, inputs[0].region_grad, logit_ip);
  //}
}

/*
  regions[0](I): x
  regions[1](O): y
*/
OpMeta *SoftmaxDP::init_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SoftmaxDPInitParams const *softmaxDP = (SoftmaxDPInitParams *)task->args;
  AccessorRO<float, 3> const acc_x(regions[0], FID_DATA);
  AccessorWO<float, 3> const acc_y(regions[1], FID_DATA);
  Rect<3> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  SoftmaxDPMeta *m = new SoftmaxDPMeta(softmaxDP->handle);
  m->profiling_runtime = softmaxDP->profiling;
  m->batchSize = softmaxDP->batchSize;
#ifndef DISABLE_COMPUTATION
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  assert(rect_x == rect_y);
  int input_c = rect_x.hi[0] - rect_x.lo[0] + 1;
  int input_n = (rect_x.hi[1] - rect_x.lo[1] + 1) * LSTM_PER_NODE_LENGTH;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        1,
                                        1));
#endif
  return m;
}

void SoftmaxDP::init(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    SoftmaxDPInitParams initParams;
    initParams.handle = model.dnn_handlers[paraConfig.gpu[idx]];
    initParams.batchSize = model.config.batchSize;
    initParams.profiling = false;
    TaskLauncher launcher(RNN_SOFTMAXDP_INIT_TASK_ID,
                          TaskArgument(&initParams, sizeof(initParams)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x = runtime->get_logical_subregion_by_color(logit_lp, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    {
      LogicalRegion y =
          runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, WRITE_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(1, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    meta[idx] = f.get_result<OpMeta *>();
  }
}

/*
  regions[0](I): x
  regions[1](O): y
*/
void SoftmaxDP::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  SoftmaxDPMeta const *m = *((SoftmaxDPMeta **)task->args);
  AccessorRO<float, 3> const acc_x(regions[0], FID_DATA);
  AccessorWO<float, 3> const acc_y(regions[1], FID_DATA);
  Rect<3> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  float const *x_ptr = acc_x.ptr(rect_x.lo);
  float *y_ptr = acc_y.ptr(rect_y.lo);

  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m->inputTensor,
                                 x_ptr,
                                 &beta,
                                 m->inputTensor,
                                 y_ptr));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("SoftmaxDP forward time = %.2fms\n", elapsed);
  }
#ifdef PRINT_INTERMEDIATE_RESULT
  print_tensor<3, float>(y_ptr, rect_y, "softmax");
#endif
#endif
}

void SoftmaxDP::forward(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    OpMeta *mp = meta[idx];
    TaskLauncher launcher(RNN_SOFTMAXDP_FWD_TASK_ID,
                          TaskArgument(&mp, sizeof(OpMeta *)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x = runtime->get_logical_subregion_by_color(logit_lp, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    {
      LogicalRegion y =
          runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, WRITE_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(1, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }
}

__global__ void SoftmaxLossBackprop(float *input,
                                    int const *label,
                                    int vocab_size,
                                    int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    int label_idx = label[i];
    input[i * vocab_size + label_idx] -= 1.0f;
  }
}

/*
  regions[0](O): x_grad
  regions[1](I): y
  regions[2](I): labels
*/
void SoftmaxDP::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  SoftmaxDPMeta const *m = *((SoftmaxDPMeta **)task->args);
  AccessorWO<float, 3> const acc_x_grad(regions[0], FID_DATA);
  AccessorRO<float, 3> const acc_y(regions[1], FID_DATA);
  AccessorRO<int, 2> const acc_label(regions[2], FID_DATA);
  Rect<3> rect_x_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_x_grad.accessor.is_dense_arbitrary(rect_x_grad));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float *x_grad_ptr = acc_x_grad.ptr(rect_x_grad.lo);
  float const *y_ptr = acc_y.ptr(rect_y.lo);
  int const *label_ptr = acc_label.ptr(rect_label.lo);
  assert(rect_x_grad == rect_y);
  assert(rect_y.hi[1] - rect_y.lo[1] == rect_label.hi[0] - rect_label.lo[0]);
  assert(rect_y.hi[2] - rect_y.lo[2] == rect_label.hi[1] - rect_label.lo[1]);
  int num_labels = rect_label.volume();
  int vocab_size = rect_y.hi[0] - rect_y.lo[0] + 1;

  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  checkCUDA(cudaMemcpyAsync(x_grad_ptr,
                            y_ptr,
                            rect_x_grad.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
  SoftmaxLossBackprop<<<GET_BLOCKS(num_labels), CUDA_NUM_THREADS>>>(
      x_grad_ptr, label_ptr, vocab_size, num_labels);

  // Accouting for batch size in SGD
  float scalVal = 1.0f / static_cast<float>(m->batchSize);
  scale_kernel<<<GET_BLOCKS(rect_x_grad.volume()), CUDA_NUM_THREADS>>>(
      x_grad_ptr, rect_x_grad.volume(), 0.0f, scalVal);
  // checkCUDA(cublasSscal(m->handle.blas, rect_x_grad.volume(),
  //                       &scalVal, x_grad_ptr, 1));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax backward time = %.2fms\n", elapsed);
  }
#ifdef PRINT_INTERMEDIATE_RESULT
  print_tensor<3, float>(x_grad_ptr, rect_x_grad, "softmax bwd:x_grad");
  float *host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr,
                          sizeof(float) * rect_x_grad.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(host_ptr,
                       x_grad_ptr,
                       sizeof(float) * rect_x_grad.volume(),
                       cudaMemcpyDeviceToHost));
  int idx = 0;
  float loss = 0.0f;
  for (PointInRectIterator<3> it(rect_x_grad); it(); it++, idx++) {
    if (host_ptr[idx] < 0) {
      loss += -std::log(host_ptr[idx] + 1);
    }
  }
  printf("lost = %.4lf\n", loss);
  checkCUDA(cudaFreeHost(host_ptr));
#endif
#endif
}

void SoftmaxDP::backward(RnnModel const &model) {
  Context ctx = model.config.lg_ctx;
  Runtime *runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++, idx++) {
    OpMeta *mp = meta[idx];
    TaskLauncher launcher(RNN_SOFTMAXDP_BWD_TASK_ID,
                          TaskArgument(&mp, sizeof(OpMeta *)),
                          Predicate::TRUE_PRED,
                          0 /*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    {
      LogicalRegion x =
          runtime->get_logical_subregion_by_color(logit_grad_lp, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
      launcher.add_field(0, FID_DATA);
    }
    {
      LogicalRegion y =
          runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, READ_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(1, FID_DATA);
    }
    {
      LogicalRegion l =
          runtime->get_logical_subregion_by_color(label.partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(l, READ_ONLY, EXCLUSIVE, label.region));
      launcher.add_field(2, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }
}

void SoftmaxDP::update(RnnModel const &model) {}
