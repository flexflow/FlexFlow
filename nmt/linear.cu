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

#include "rnn.h"
#include "rnn_mapper.h"
#include "../cnn_helper.h"

struct LinearInitParams {
  DnnHandle handle;
  int batchSize, inputSize, outputSize;
};

Tensor RnnModel::add_linear_node(Tensor x, int output_size,
                                 ParallelConfig pc, SharedVariable params)
{
  assert(x.numDim == 3);
  assert(x.adim[2] == LSTM_PER_NODE_LENGTH);
  assert(x.pdim[2] == LSTM_PER_NODE_LENGTH);
  Linear* node = new Linear(config, x, output_size, pc, params, part_is);
  layers.push_back(node);
  return node->outputs[0];
}

Linear::Linear(RnnConfig config, Tensor input, int _output_size,
               ParallelConfig pc, SharedVariable _params,
               IndexSpaceT<1> input_part_is)
: RnnOp(input, pc, _params), input_size(input.adim[0]), output_size(_output_size)
{
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;
  assert(pc.nDims == 2);
  int num_par_n = pc.dim[1];
  int num_par_c = pc.dim[0];
  {
    Rect<2> rect(Point<2>(0, 0), Point<2>(num_par_c-1, num_par_n-1));
    part_rect = rect;
  }
  IndexSpaceT<2> part_is = runtime->create_index_space(ctx, part_rect);
  int batch_size = input.adim[1];
  FieldSpace fs = config.field_space;
  Rect<3, coord_t> y_rect(Point<3>(0, 0, 0),
                          Point<3>(output_size-1, batch_size-1, LSTM_PER_NODE_LENGTH-1));
  IndexSpaceT<3> y_is = runtime->create_index_space(ctx, y_rect);
  LogicalRegion y_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion y_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  assert(output_size % num_par_c == 0);
  assert(batch_size % num_par_n == 0);
  int extent_c = output_size / num_par_c;
  int extent_n = batch_size / num_par_n;
  Rect<3, coord_t> extent(Point<3>(0, 0, 0),
                          Point<3>(extent_c-1, extent_n-1, LSTM_PER_NODE_LENGTH-1));
  Transform<3, 2, coord_t> trans;
  trans[0][0] = extent_c; trans[0][1] = 0;
  trans[1][0] = 0; trans[1][1] = extent_n;
  trans[2][0] = 0; trans[2][1] = 0;
  IndexPartition y_ip =
    runtime->create_partition_by_restriction(ctx, y_is, part_is, trans, extent);
  assert(runtime->is_index_partition_disjoint(ctx, y_ip));
  assert(runtime->is_index_partition_complete(ctx, y_ip));
  LogicalPartition y_lp = runtime->get_logical_partition(ctx, y_lr, y_ip);
  LogicalPartition y_grad_lp = runtime->get_logical_partition(ctx, y_grad_lr, y_ip);

  // Note: we only need replica's grad, so no need to create lr/lp for forward
  Rect<3, coord_t> replica_rect(Point<3>(0, 0, 0),
                                Point<3>(input_size-1, batch_size-1,
                                         LSTM_PER_NODE_LENGTH*num_par_c-1));
  IndexSpaceT<3> replica_is = runtime->create_index_space(ctx, replica_rect);
  replica.region_grad = runtime->create_logical_region(ctx, replica_is, fs);
  trans[0][0] = 0; trans[0][1] = 0;
  trans[1][0] = 0; trans[1][1] = extent_n;
  trans[2][0] = LSTM_PER_NODE_LENGTH; trans[2][1] = 0;
  Rect<3, coord_t> replica_ext(Point<3>(0, 0, 0),
                               Point<3>(input_size-1, extent_n-1, LSTM_PER_NODE_LENGTH-1));
  IndexPartition replica_ip =
    runtime->create_partition_by_restriction(ctx, replica_is, part_is, trans, replica_ext);
  assert(runtime->is_index_partition_disjoint(ctx, replica_ip));
  assert(runtime->is_index_partition_complete(ctx, replica_ip));
  replica.partition_grad = runtime->get_logical_partition(ctx, replica.region_grad, replica_ip);
  for (int i = 0; i < num_par_c; i++) {
    trans[0][0] = inputs[0].pdim[0]; trans[0][1] = 0;
    trans[1][0] = 0; trans[1][1] = inputs[0].pdim[1];
    trans[2][0] = 0; trans[2][1] = 0;
    Rect<3, coord_t> ext(Point<3>(0, 0, LSTM_PER_NODE_LENGTH*i),
                         Point<3>(inputs[0].pdim[0]-1, inputs[0].pdim[1]-1,
                                  LSTM_PER_NODE_LENGTH*(i+1)-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, replica_is, part_is, trans, ext);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    replica_sub_lps[i] = runtime->get_logical_partition(ctx, replica.region_grad, ip);
  }

  outputs[0].numDim = 3;
  outputs[0].adim[0] = output_size;
  outputs[0].adim[1] = batch_size;
  outputs[0].adim[2] = LSTM_PER_NODE_LENGTH;
  outputs[0].pdim[0] = extent_c;
  outputs[0].pdim[1] = extent_n;
  outputs[0].pdim[2] = LSTM_PER_NODE_LENGTH;
  outputs[0].region = y_lr;
  outputs[0].partition = y_lp;
  outputs[0].region_grad = y_grad_lr;
  outputs[0].partition_grad = y_grad_lp;

  // Every partition reads all in_channels
  trans[0][0] = 0; trans[0][1] = 0;
  trans[1][0] = 0; trans[1][1] = 0;
  trans[2][0] = 0; trans[2][1] = 0;
  Rect<3, coord_t> input_ext(Point<3>(0, 0, 0),
                             Point<3>(input_size-1, extent_n-1, LSTM_PER_NODE_LENGTH));
  IndexSpaceT<3> input_is = IndexSpaceT<3>(inputs[0].region.get_index_space());
  IndexPartition input_ip
    = runtime->create_partition_by_restriction(ctx, input_is, part_is, trans, input_ext);
  input_lp = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
}

/*
  regions[0](I): x
  regions[1](I): w
  regions[2](O): y
 */
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const LinearInitParams* linear = (LinearInitParams*) task->args;
  Rect<3> rect_x =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  assert(rect_x.hi[0] - rect_x.lo[0] + 1 == linear->inputSize);
  assert(rect_x.hi[1] - rect_x.lo[1] + 1 == linear->batchSize);
  assert(rect_x.hi[2] - rect_x.lo[2] + 1 == LSTM_PER_NODE_LENGTH);
  assert(rect_y.hi[0] - rect_y.lo[0] + 1 == linear->outputSize);
  assert(rect_y.hi[1] - rect_y.lo[1] + 1 == linear->batchSize);
  assert(rect_y.hi[2] - rect_y.lo[2] + 1 == LSTM_PER_NODE_LENGTH);
  assert(rect_w.hi[0] - rect_w.lo[0] + 1 == linear->outputSize*(linear->inputSize+1));
  LinearMeta* m = new LinearMeta(linear->handle);
#ifndef DISABLE_COMPUTATION
  int batch_size = linear->batchSize;
  float* dram_one_ptr = (float*) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  checkCUDA(cudaMalloc(&m->one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(m->one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyHostToDevice));
#endif
  return m;
}

void Linear::init(const RnnModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<2> it(part_rect); it(); it++, idx++) {
    LinearInitParams initParams;
    initParams.handle = model.dnn_handlers[idx];
    initParams.batchSize = outputs[0].pdim[1];
    initParams.inputSize = inputs[0].pdim[0];
    initParams.outputSize = outputs[0].pdim[0];
    assert(outputs[0].pdim[1] == inputs[0].pdim[1]);
    TaskLauncher launcher(RNN_LINEAR_INIT_TASK_ID,
                          TaskArgument(&initParams, sizeof(initParams)),
                          Predicate::TRUE_PRED, 0/*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    // Add input
    {
      LogicalRegion x = runtime->get_logical_subregion_by_color(input_lp, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.subregions[dp[0]], READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(1, FID_DATA);
    // Add output
    {
      LogicalRegion y =
        runtime->get_logical_subregion_by_color(outputs[0].partition, dp);
      launcher.add_region_requirement(
          RegionRequirement(y, WRITE_ONLY, EXCLUSIVE, outputs[0].region));
      launcher.add_field(2, FID_DATA);
    }
    Future f = runtime->execute_task(ctx, launcher);
    meta[idx] = f.get_result<OpMeta*>();
  }
}

/*
  regions[0] (I): x
  regions[1] (I): w
  regions[2] (O): y
 */
void Linear::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  float alpha = 1.0f, beta = 0.0f;
  const LinearMeta* m = *((LinearMeta**) task->args);
  const AccessorRO<float, 3> acc_x(regions[0], FID_DATA);
  const AccessorRO<float, 1> acc_w(regions[1], FID_DATA);
  const AccessorWO<float, 3> acc_y(regions[2], FID_DATA);
  Rect<3> rect_x =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  int input_size = rect_x.hi[0] - rect_x.lo[0] + 1;
  int output_size = rect_y.hi[0] - rect_y.lo[0] + 1;
  int batch_size = rect_x.hi[1] - rect_x.lo[1] + 1;
  const float *x_ptr = acc_x.ptr(rect_x.lo);
  const float *w_ptr = acc_w.ptr(rect_w.lo);
  const float *bias_ptr = w_ptr + input_size;
  float *y_ptr = acc_y.ptr(rect_y.lo);
  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        output_size, batch_size, input_size,
                        &alpha, w_ptr, input_size + 1,
                        x_ptr, input_size, &beta,
                        y_ptr, output_size));
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        output_size, batch_size, 1,
                        &alpha, bias_ptr, input_size + 1,
                        m->one_ptr, 1, &alpha,
                        y_ptr, output_size));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear forward time = %.2lfms\n", elapsed);
  }
#endif
}

void Linear::forward(const RnnModel &model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<2> it(part_rect); it(); it++, idx++) {
    OpMeta* mp = meta[idx];
    TaskLauncher launcher(RNN_LINEAR_FWD_TASK_ID,
                          TaskArgument(&mp, sizeof(OpMeta*)),
                          Predicate::TRUE_PRED, 0/*MapperID*/,
                          RnnMapper::assign_to_gpu(paraConfig.gpu[idx]));
    DomainPoint dp(*it);
    // Add input
    {
      LogicalRegion x = runtime->get_logical_subregion_by_color(input_lp, dp);
      launcher.add_region_requirement(
          RegionRequirement(x, READ_ONLY, EXCLUSIVE, inputs[0].region));
      launcher.add_field(0, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.subregions[dp[0]], READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(1, FID_DATA);
    // Add output
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
  regions[1](I): w
  regions[2](I): y
  regions[3](O); replica_grad
  regions[4](I/O): w_grad
  regions[5](I): y_grad
*/
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  float alpha = 1.0f, beta = 0.0f;
  const LinearMeta* m = *((LinearMeta**) task->args);
  const AccessorRO<float, 3> acc_x(regions[0], FID_DATA);
  const AccessorRO<float, 1> acc_w(regions[1], FID_DATA);
  const AccessorRO<float, 3> acc_y(regions[2], FID_DATA);
  const AccessorWO<float, 3> acc_replica_grad(regions[3], FID_DATA);
  const AccessorRW<float, 1> acc_w_grad(regions[4], FID_DATA);
  const AccessorRO<float, 3> acc_y_grad(regions[5], FID_DATA);

  Rect<3> rect_x =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_w =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_y =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  Rect<3> rect_replica_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_w_grad =
    runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  Rect<3> rect_y_grad =
    runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  assert(acc_replica_grad.accessor.is_dense_arbitrary(rect_replica_grad));
  assert(acc_w_grad.accessor.is_dense_arbitrary(rect_w_grad));
  assert(acc_y_grad.accessor.is_dense_arbitrary(rect_y_grad));
  int input_size = rect_x.hi[0] - rect_x.lo[0] + 1;
  int output_size = rect_y.hi[0] - rect_y.lo[0] + 1;
  int batch_size = rect_x.hi[1] - rect_x.lo[1] + 1;
  const float *x_ptr = acc_x.ptr(rect_x.lo);
  const float *w_ptr = acc_w.ptr(rect_w.lo);
  const float *y_ptr = acc_y.ptr(rect_y.lo);
  float* replica_grad_ptr = acc_replica_grad.ptr(rect_replica_grad.lo);
  float* w_grad_ptr = acc_w_grad.ptr(rect_w_grad.lo);
  float* bias_grad_ptr = w_grad_ptr + input_size;
  const float* y_grad_ptr = acc_y_grad.ptr(rect_y_grad.lo);
  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  // Compute weight gradient
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        input_size, output_size, batch_size,
                        &alpha, x_ptr, input_size,
                        y_grad_ptr, output_size,
                        &beta, w_grad_ptr, input_size+1));
  // Compute bias gradient
  checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                        output_size, batch_size,
                        &alpha, y_grad_ptr, output_size,
                        m->one_ptr, 1,
                        &beta, bias_grad_ptr, input_size+1));
  // Compute data gradient
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        input_size, batch_size, output_size,
                        &alpha, w_ptr, input_size,
                        y_grad_ptr, output_size,
                        &beta, replica_grad_ptr, input_size));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear backward time = %.2lfms\n", elapsed);
  }
#endif
}

void Linear::backward(const RnnModel& model)
{}

void Linear::update_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{}

void Linear::update(const RnnModel& model)
{
}
