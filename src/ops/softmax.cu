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

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::softmax(std::string name, Tensor input)
{
  assert(input.numDim == 2);
  IndexSpaceT<1> task_is;
  Softmax *sm = new Softmax(name, config, input, task_is);
  layers.push_back(sm);
  return sm->output;
}

Softmax::Softmax(std::string _name, FFConfig _config,
                 Tensor _input, IndexSpaceT<1> _task_is)
: Op(_name, _input), task_is(_task_is), profiling(_config.profiling)
{
  assert(_input.numDim == 2);
  Context ctx = _config.lg_ctx;
  HighLevelRuntime* runtime = _config.lg_hlr;
  Rect<1> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_n = part_rect.hi[0] - part_rect.lo[0] + 1;

  FieldSpace fs = _config.field_space;
  IndexSpaceT<2> output_is(_input.region.get_index_space());
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<2, 1, coord_t> transform;
  int extent_c = _input.adim[0];
  int extent_n = (_input.adim[1] + num_par_n - 1) / num_par_n;
  Rect<2, coord_t> extent(Point<2>(0, 0), Point<2>(extent_c-1, extent_n-1));
  transform[0][0] = 0;
  transform[1][0] = extent_n;
  IndexPartition output_ip
    = runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, output_ip));
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  output.numDim = 2;
  output.adim[0] = _input.adim[0];
  output.adim[1] = _input.adim[1];
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.part = output_lp;
  // Note: we use the same regions/partitions for forward and back prop
  output.region_grad = output_lr;
  output.part_grad = output_lp;
  // Every partition reads all input_channels
  // Use the same partitioning as outputs
  IndexPartition input_ip = output_ip;
  input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
  input_grad_lp = runtime->get_logical_partition(ctx, inputs[0].region_grad, input_ip);
}

/*
  regions[0]: input
  regions[1]: output
 */
OpMeta* Softmax::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Softmax* softmax = (Softmax*) task->args;
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  Rect<2> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  FFHandler handle = *((const FFHandler*) task->local_args);
  SoftmaxMeta* m = new SoftmaxMeta(handle);
#ifndef DISABLE_COMPUTATION
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  //checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  assert(rect_input == rect_output);
  int input_c = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_n = rect_input.hi[1] - rect_input.lo[1] + 1;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, 1, 1));
#endif
  return m;
}

__host__
void Softmax::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher init_launcher(SOFTMAX_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(Softmax)), argmap);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
__host__
void Softmax::forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  Rect<2> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);

  cudaEvent_t t_start, t_end;
  if (softmax->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->inputTensor, output_ptr));
  if (softmax->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax forward time = %.2fms\n", elapsed);
  }
#endif
}

__host__
void Softmax::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<1> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

__global__ void SoftmaxLossBackprop(float *input, const int *label, int num_labels, int batch_size)
{
  CUDA_KERNEL_LOOP(i, batch_size)
  {
    int label_idx = label[i];
    input[i * num_labels + label_idx] -= 1.0f;
  }
}

/*
  regions[0](O): input_grad
  regions[1](I): output
  regions[2](I): labels
*/
__host__
void Softmax::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  const AccessorRW<float, 2> acc_input_grad(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<int, 1> acc_label(regions[2], FID_DATA);
  Rect<2> rect_input_grad, rect_output;
  Rect<1> rect_label;
  rect_input_grad =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_label =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  assert(acc_input_grad.accessor.is_dense_arbitrary(rect_input_grad));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  // make sure the image indices match!
  assert(rect_label.hi[0] == rect_output.hi[1]);
  assert(rect_label.lo[0] == rect_output.lo[1]);
  int num_images = rect_label.volume();
  int num_labels = rect_output.hi[0] - rect_output.lo[0] + 1;
  assert(num_labels == 1000); // check that we have 1000 different labels

  float *input_grad_ptr = acc_input_grad.ptr(rect_input_grad.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  const int *label_ptr = acc_label.ptr(rect_label.lo);

  cudaEvent_t t_start, t_end;
  if (softmax->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDA(cudaMemcpyAsync(input_grad_ptr, output_ptr,
                            rect_input_grad.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
  SoftmaxLossBackprop<<<GET_BLOCKS(num_images), CUDA_NUM_THREADS>>>(
      input_grad_ptr, label_ptr, num_labels, num_images);

  // Accouting for batch size in SGD
  float scalVal = 1.0f / static_cast<float>(num_images);
  checkCUDA(cublasSscal(m->handle.blas, rect_input_grad.volume(),
                        &scalVal, input_grad_ptr, 1));
  if (softmax->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax backward time = %.2fms\n", elapsed);
  }
#endif
}

__host__
void Softmax::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  int idx = 0;
  Rect<1> rect = runtime->get_index_space_domain(ctx, task_is);
  for (PointInRectIterator<1> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lp, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
#ifdef DEADCODE
  launcher.add_region_requirement(
      RegionRequirement(ff.inputLabel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, ff.inputLabel.region));
  launcher.add_field(2, FID_DATA);
#endif
  runtime->execute_index_space(ctx, launcher);
}

//void Softmax::update(const FFModel& ff)
//{
//}
