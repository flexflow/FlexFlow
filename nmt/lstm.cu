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
#include "../cnn_helper.h"

LSTMTensors RnnModel::add_lstm_node(Tensor x, Tensor hx, Tensor cx, SharedVariable params)
{
  assert(x.numDim == 2);
  assert(hx.numDim == 2);
  assert(cx.numDim == 2);
  int batch_size = x.adim[1];
  assert(hx.adim[1] == batch_size);
  assert(cx.adim[1] == batch_size);
  int input_size = x.adim[0];
  int output_size = hx.adim[0];
  assert(cx.adim[0] == output_size);
  LSTM* node = new LSTM(config, x, hx, cx, part_is,
                        batch_size, input_size, output_size);
  layers.push_back(node);
  LSTMTensors output;
  output.x = node->outputs[0];
  output.hx = node->outputs[1];
  output.cx = node->outputs[2];
  return output;
}

/*
 output[0]: y
 output[1]: hy
 output[2]: cy
 */
LSTM::LSTM(RnnConfig config, Tensor x, Tensor hx, Tensor cx,
           IndexSpaceT<1> part_is,
           int _batch_size, int _input_size, int _output_size)
: RnnOp(x, hx, cx), batch_size(_batch_size),
  input_size(_input_size), output_size(_output_size)
{
  printf("LSTM node: batch(%d) input(%d) output(%d)\n",
         batch_size, input_size, output_size);
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;
  part_rect = runtime->get_index_space_domain(ctx, part_is);
  FieldSpace fs = config.field_space;
  Rect<2, coord_t> y_rect(Point<2>(0, 0),
                          Point<2>(output_size-1, batch_size-1));
  IndexSpaceT<2> y_is = runtime->create_index_space(ctx, y_rect);
  LogicalRegion y_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion y_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  int num_par_n = part_rect.hi[0] - part_rect.lo[0] + 1;
  assert(batch_size % num_par_n == 0);
  int extent_n = batch_size / num_par_n;
  int extent_c = output_size;
  Rect<2, coord_t> extent(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  Transform<1, 2, coord_t> trans;
  trans[0][0] = 0; trans[0][1] = extent_n;
  IndexPartition y_ip =
    runtime->create_partition_by_restriction(ctx, y_is, part_is, trans, extent);
  LogicalPartition y_lp = runtime->get_logical_partition(ctx, y_lr, y_ip);
  LogicalPartition y_grad_lp = runtime->get_logical_partition(ctx, y_grad_lr, y_ip);
  outputs[0].region = y_lr;
  outputs[0].region_grad = y_grad_lr;
  outputs[0].partition = y_lp;
  outputs[0].partition_grad = y_grad_lp;
  outputs[0].numDim = 2;
  outputs[0].adim[0] = output_size;
  outputs[0].adim[1] = batch_size;
  outputs[0].pdim[0] = extent_c;
  outputs[0].pdim[1] = extent_n;

  LogicalRegion hy_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion hy_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalPartition hy_lp = runtime->get_logical_partition(ctx, hy_lr, y_ip);
  LogicalPartition hy_grad_lp = runtime->get_logical_partition(ctx, hy_grad_lr, y_ip);
  outputs[1] = outputs[0];
  outputs[1].region = hy_lr;
  outputs[1].region_grad = hy_grad_lr;
  outputs[1].partition = hy_lp;
  outputs[1].partition_grad = hy_grad_lp;

  LogicalRegion cy_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalRegion cy_grad_lr = runtime->create_logical_region(ctx, y_is, fs);
  LogicalPartition cy_lp = runtime->get_logical_partition(ctx, cy_lr, y_ip);
  LogicalPartition cy_grad_lp = runtime->get_logical_partition(ctx, cy_grad_lr, y_ip);
  outputs[2] = outputs[0];
  outputs[2].region = cy_lr;
  outputs[2].region_grad = cy_grad_lr;
  outputs[2].partition = cy_lp;
  outputs[2].partition_grad = cy_grad_lp;
}

/*
  regions[0] (I): x
  regions[1] (I): hx
  regions[2] (I): cx
  regions[3] (I): w
  regions[4] (O): y
  regions[5] (O): hy
  regions[6] (O): cy
*/
OpMeta* LSTM::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  const int numLayers = 1;
  const int seqLength = 1;
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  Rect<1> para_rect =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  const LSTM* lstm = (LSTM*) task->args;
  DnnHandle handle = *((const DnnHandle*) task->local_args);
  LSTMMeta* m = new LSTMMeta(handle);
#ifndef DISABLE_COMPUTATION
  checkCUDNN(cudnnCreateRNNDescriptor(&m->rnnDesc));
  checkCUDNN(cudnnCreateDropoutDescriptor(&m->dropoutDesc));
  size_t dropoutSize;
  void *dropoutStates;
  checkCUDNN(cudnnDropoutGetStatesSize(m->handle.dnn, &dropoutSize));
  checkCUDA(cudaMalloc(&dropoutStates, dropoutSize));
  checkCUDNN(cudnnSetRNNDescriptor(m->rnnDesc, lstm->output_size, numLayers, m->dropoutDesc,
                                   CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
                                   CUDNN_DATA_FLOAT));
  for (int i = 0; i < seqLength; i++) {
    checkCUDNN(cudnnCreateTensorDescriptor(&m->xDescs[i]));
    int dims[] = {lstm->batch_size, lstm->input_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(m->xDescs[i], CUDNN_DATA_FLOAT, 
                                          3, dims, strides));
  }
  size_t workSpaceSize;
  checkCUDNN(cudnnGetRNNWorkspaceSize(m->handle.dnn, m->rnnDesc, seqLength,
                                      m->xDescs, &workSpaceSize));
  // Assert that we have enough work space
  assert(workSpaceSize <= m->handle.workSpaceSize);
  checkCUDNN(cudnnGetRNNTrainingReserveSize(m->handle.dnn, m->rnnDesc, seqLength,
                                            m->xDescs, &m->reserveSpaceSize));
  checkCUDA(cudaMalloc(&m->reserveSpace, m->reserveSpaceSize));
  size_t paramsSize;
  checkCUDNN(cudnnGetRNNParamsSize(m->handle.dnn, m->rnnDesc, m->xDescs[0],
                                   &paramsSize, CUDNN_DATA_FLOAT));
  assert(paramsSize == para_rect.volume());
  {
    int dims[] = {(int)paramsSize, 1, 1};
    checkCUDNN(cudnnCreateFilterDescriptor(&m->wDesc));
    checkCUDNN(cudnnSetFilterNdDescriptor(m->wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          sizeof(dims) / sizeof(dims[0]), dims));
  }
  {
    checkCUDNN(cudnnCreateTensorDescriptor(&m->hxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&m->cxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&m->hyDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&m->cyDesc));
    int dims[] = {numLayers, lstm->batch_size, lstm->output_size};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(m->hxDesc, CUDNN_DATA_FLOAT,
                                          3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(m->cxDesc, CUDNN_DATA_FLOAT,
                                          3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(m->hyDesc, CUDNN_DATA_FLOAT,
                                          3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(m->cyDesc, CUDNN_DATA_FLOAT,
                                          3, dims, strides));
  }
  for (int i = 0; i < seqLength; i++) {
    checkCUDNN(cudnnCreateTensorDescriptor(&m->yDescs[i]));
    int dims[] = {lstm->batch_size, lstm->output_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(m->yDescs[i], CUDNN_DATA_FLOAT,
                                          3, dims, strides));
  }
  return m;
#endif
}

void LSTM::init(const RnnModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
}

/*
  regions[0] (I): x
  regions[1] (I): hx
  regions[2] (I): cx
  regions[3] (I): w
  regions[4] (O): y
  regions[5] (O): hy
  regions[6] (O): cy
*/
void LSTM::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  const LSTMMeta* m = *((LSTMMeta**) task->args);
  const AccessorRO<float, 2> acc_x(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_hx(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_cx(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_w(regions[3], FID_DATA);
  const AccessorWO<float, 2> acc_y(regions[4], FID_DATA);
  const AccessorWO<float, 2> acc_hy(regions[5], FID_DATA);
  const AccessorWO<float, 2> acc_cy(regions[6], FID_DATA);
  Rect<2> rect_x, rect_hx, rect_cx, rect_y, rect_hy, rect_cy;
  Rect<1> rect_w;
  rect_x = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_hx = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_cx = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_w = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_y = runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  rect_hy = runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  rect_cy = runtime->get_index_space_domain(ctx, task->regions[6].region.get_index_space());
  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_hx.accessor.is_dense_arbitrary(rect_hx));
  assert(acc_cx.accessor.is_dense_arbitrary(rect_cx));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  assert(acc_hy.accessor.is_dense_arbitrary(rect_hy));
  assert(acc_cy.accessor.is_dense_arbitrary(rect_cy));
  assert(rect_hx == rect_cx);
  assert(rect_hx == rect_y);
  assert(rect_hx == rect_hy);
  assert(rect_hx == rect_cy);
  const float *x_ptr = acc_x.ptr(rect_x.lo);
  const float *hx_ptr = acc_hx.ptr(rect_hx.lo);
  const float *cx_ptr = acc_cx.ptr(rect_cx.lo);
  const float *w_ptr = acc_w.ptr(rect_w.lo);
  float *y_ptr = acc_y.ptr(rect_y.lo);
  float *hy_ptr = acc_hy.ptr(rect_hy.lo);
  float *cy_ptr = acc_cy.ptr(rect_cy.lo);
  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  checkCUDNN(cudnnRNNForwardTraining(m->handle.dnn, m->rnnDesc, 1/*seqLength*/,
                                     m->xDescs, x_ptr, m->hxDesc, hx_ptr,
                                     m->cxDesc, cx_ptr, m->wDesc, w_ptr,
                                     m->yDescs, y_ptr, m->hyDesc, hy_ptr,
                                     m->cyDesc, cy_ptr,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     m->reserveSpace, m->reserveSpaceSize));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("LSTM forward time = %.2fms\n", elapsed);
  }
#endif
}

void LSTM::forward(const RnnModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    TaskLauncher launcher(LSTM_FWD_TASK_ID, TaskArgument(&mp, sizeof(OpMeta*)));
    DomainPoint dp(*it);
    // add region requirements for x, hx, cx
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(inputs[i].partition, dp);
      launcher.add_region_requirement(RegionRequirement(x, READ_ONLY, EXCLUSIVE, x));
      launcher.add_field(i, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.region, READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(3, FID_DATA);
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(outputs[i].partition, dp);
      launcher.add_region_requirement(RegionRequirement(x, WRITE_ONLY, EXCLUSIVE, x));
      launcher.add_field(4 + i, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
  }
}

/*
  regions[0] (I): x
  regions[1] (I): hx
  regions[2] (I): cx
  regions[3] (I): w
  regions[4] (I): y
  regions[5] (I): hy
  regions[6] (I): cy
  regions[7] (O): x_grad
  regions[8] (O): hx_grad
  regions[9] (O): cx_grad
 regions[10] (I/O): w_grad
 regions[11] (I): y_grad
 regions[12] (I): hy_grad
 regions[13] (I): cy_grad
*/
void LSTM::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 14);
  assert(task->regions.size() == 14);
  const LSTMMeta* m = *((LSTMMeta**) task->args);
  const AccessorRO<float, 2> acc_x(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_hx(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_cx(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_w(regions[3], FID_DATA);
  const AccessorRO<float, 2> acc_y(regions[4], FID_DATA);
  const AccessorRO<float, 2> acc_hy(regions[5], FID_DATA);
  const AccessorRO<float, 2> acc_cy(regions[6], FID_DATA);
  const AccessorWO<float, 2> acc_x_grad(regions[7], FID_DATA);
  const AccessorWO<float, 2> acc_hx_grad(regions[8], FID_DATA);
  const AccessorWO<float, 2> acc_cx_grad(regions[9], FID_DATA);
  const AccessorRW<float, 1> acc_w_grad(regions[10], FID_DATA);
  const AccessorRO<float, 2> acc_y_grad(regions[11], FID_DATA);
  const AccessorRO<float, 2> acc_hy_grad(regions[12], FID_DATA);
  const AccessorRO<float, 2> acc_cy_grad(regions[13], FID_DATA);

  Rect<2> rect_x, rect_hx, rect_cx, rect_y, rect_hy, rect_cy,
          rect_x_grad, rect_hx_grad, rect_cx_grad,
          rect_y_grad, rect_hy_grad, rect_cy_grad;
  Rect<1> rect_w, rect_w_grad;
  rect_x =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_hx =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_cx =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_w =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_y =
    runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  rect_hy =
    runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  rect_cy =
    runtime->get_index_space_domain(ctx, task->regions[6].region.get_index_space());
  rect_x_grad =
    runtime->get_index_space_domain(ctx, task->regions[7].region.get_index_space());
  rect_hx_grad =
    runtime->get_index_space_domain(ctx, task->regions[8].region.get_index_space());
  rect_cx_grad =
    runtime->get_index_space_domain(ctx, task->regions[9].region.get_index_space());
  rect_w_grad =
    runtime->get_index_space_domain(ctx, task->regions[10].region.get_index_space());
  rect_y_grad =
    runtime->get_index_space_domain(ctx, task->regions[11].region.get_index_space());
  rect_hy_grad =
    runtime->get_index_space_domain(ctx, task->regions[12].region.get_index_space());
  rect_cy_grad =
    runtime->get_index_space_domain(ctx, task->regions[13].region.get_index_space());

  assert(acc_x.accessor.is_dense_arbitrary(rect_x));
  assert(acc_hx.accessor.is_dense_arbitrary(rect_hx));
  assert(acc_cx.accessor.is_dense_arbitrary(rect_cx));
  assert(acc_w.accessor.is_dense_arbitrary(rect_w));
  assert(acc_y.accessor.is_dense_arbitrary(rect_y));
  assert(acc_hy.accessor.is_dense_arbitrary(rect_hy));
  assert(acc_cy.accessor.is_dense_arbitrary(rect_cy));
  assert(acc_x_grad.accessor.is_dense_arbitrary(rect_x_grad));
  assert(acc_hx_grad.accessor.is_dense_arbitrary(rect_hx_grad));
  assert(acc_cx_grad.accessor.is_dense_arbitrary(rect_cx_grad));
  assert(acc_w_grad.accessor.is_dense_arbitrary(rect_w_grad));
  assert(acc_y_grad.accessor.is_dense_arbitrary(rect_y_grad));
  assert(acc_hy_grad.accessor.is_dense_arbitrary(rect_hy_grad));
  assert(acc_cy_grad.accessor.is_dense_arbitrary(rect_cy_grad));

  const float *x_ptr = acc_x.ptr(rect_x.lo);
  const float *hx_ptr = acc_hx.ptr(rect_hx.lo);
  const float *cx_ptr = acc_cx.ptr(rect_cx.lo);
  const float *w_ptr = acc_w.ptr(rect_w.lo);
  const float *y_ptr = acc_y.ptr(rect_y.lo);
  const float *hy_ptr = acc_hy.ptr(rect_hy.lo);
  const float *cy_ptr = acc_cy.ptr(rect_cy.lo);
  float *x_grad_ptr = acc_x_grad.ptr(rect_x_grad.lo);
  float *hx_grad_ptr = acc_hx_grad.ptr(rect_hx_grad.lo);
  float *cx_grad_ptr = acc_cx_grad.ptr(rect_cx_grad.lo);
  float *w_grad_ptr = acc_w_grad.ptr(rect_w_grad.lo);
  const float *y_grad_ptr = acc_y_grad.ptr(rect_y_grad.lo);
  const float *hy_grad_ptr = acc_hy_grad.ptr(rect_hy_grad.lo);
  const float *cy_grad_ptr = acc_cy_grad.ptr(rect_cy_grad.lo);

  cudaEvent_t t_start, t_end;
  if (m->profiling_runtime) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  checkCUDNN(cudnnRNNBackwardData(m->handle.dnn, m->rnnDesc, 1/*seqLength*/,
                                  m->yDescs, y_ptr, m->yDescs, y_grad_ptr,
                                  m->hyDesc, hy_grad_ptr, m->cyDesc, cy_grad_ptr,
                                  m->wDesc, w_ptr, m->hxDesc, hx_ptr,
                                  m->cxDesc, cx_ptr, m->xDescs, x_grad_ptr,
                                  m->hxDesc, hx_grad_ptr, m->cxDesc, cx_grad_ptr,
                                  m->handle.workSpace, m->handle.workSpaceSize,
                                  m->reserveSpace, m->reserveSpaceSize));
  checkCUDNN(cudnnRNNBackwardWeights(m->handle.dnn, m->rnnDesc, 1/*seqLength*/,
                                    m->xDescs, x_ptr, m->hxDesc, hx_ptr,
                                    m->yDescs, y_ptr,
                                    m->handle.workSpace, m->handle.workSpaceSize,
                                    m->wDesc, w_grad_ptr,
                                    m->reserveSpace, m->reserveSpaceSize));
  if (m->profiling_runtime) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("LSTM backward time = %.2fms\n", elapsed);
  }
#endif
}

void LSTM::backward(const RnnModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  int idx = 0;
  for (PointInRectIterator<1> it(part_rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    DomainPoint dp(*it);
    TaskLauncher launcher(LSTM_BWD_TASK_ID, TaskArgument(&mp, sizeof(OpMeta*)));
    // add region requirements for x, hx, cx
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(inputs[i].partition, dp);
      launcher.add_region_requirement(RegionRequirement(x, READ_ONLY, EXCLUSIVE, x));
      launcher.add_field(i, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.region, READ_ONLY, EXCLUSIVE, params.region));
    launcher.add_field(3, FID_DATA);
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(outputs[i].partition, dp);
      launcher.add_region_requirement(RegionRequirement(x, READ_ONLY, EXCLUSIVE, x));
      launcher.add_field(4 + i, FID_DATA);
    }
    // add region requirements for gradients
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(inputs[i].partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(x, WRITE_ONLY, EXCLUSIVE, x));
      launcher.add_field(7+i, FID_DATA);
    }
    launcher.add_region_requirement(
        RegionRequirement(params.gradients[0], READ_WRITE, EXCLUSIVE, params.gradients[0]));
    launcher.add_field(10, FID_DATA);
    for (int i = 0; i < 3; i++) {
      LogicalRegion x =
        runtime->get_logical_subregion_by_color(outputs[i].partition_grad, dp);
      launcher.add_region_requirement(RegionRequirement(x, READ_ONLY, EXCLUSIVE, x));
      launcher.add_field(11 + i, FID_DATA);
    }
  }
}

void LSTM::update(const RnnModel& model)
{
}
