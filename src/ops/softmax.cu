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


Tensor FFModel::softmax(const Tensor& _input)
{
  assert(_input.numDim == 2);
  Softmax *sm = new Softmax(*this, _input);
  layers.push_back(sm);
  return sm->outputs[0];
}

Softmax::Softmax(FFModel& model,
                 const Tensor& _input)
: Op(model, OP_SOFTMAX, "Softmax", _input), profiling(model.config.profiling)
{
  outputs[0].numDim = 2;
  outputs[0].adim[0] = _input.adim[0];
  outputs[0].adim[1] = _input.adim[1];
}


void Softmax::create_weights(FFModel& model)
{
  // Do nothing since we don't ahve weights
}

void Softmax::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  // Current require data parallelism for Softmax
  assert(num_par_c == 1);
  {
    const int dims[2] = {inputs[0].adim[1], inputs[0].adim[0]};
    outputs[0] = model.create_tensor<2>(dims, (IndexSpaceT<2>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition(
        inputs[0], (IndexSpaceT<2>)task_is, input_lps[0], input_grad_lps[0]);
  }
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
  TensorAccessorR<float, 2> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readutput*/);
  FFHandler handle = *((const FFHandler*) task->local_args);
  SoftmaxMeta* m = new SoftmaxMeta(handle);
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  //checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  assert(acc_input.rect == acc_output.rect);
  int input_c = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_n = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, 1, 1));
  return m;
}

__host__
void Softmax::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(SOFTMAX_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
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
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  TensorAccessorR<float, 2> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

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
                                 &alpha, m->inputTensor, acc_input.ptr,
                                 &beta, m->inputTensor, acc_output.ptr));
  if (softmax->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<2, float>(acc_output.ptr, acc_output.rect, "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax forward time = %.2fms\n", elapsed);
  }
}

__host__
void Softmax::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/
// Note that the backward task of softmax is actually a no op (i.e., input_grad = output_grad)
// since the upstream cross_entropy_loss function computes performs softmax_cross_entropy_loss
// to avoid intermediate zeros
__host__
void Softmax::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  TensorAccessorW<float, 2> acc_input_grad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 2> acc_output_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // make sure the image indices match!
  assert(acc_input_grad.rect == acc_output_grad.rect);

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
  checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
                            acc_input_grad.rect.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
  if (softmax->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<2, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Softmax:backward:output_grad]");
    //print_tensor<2, float>(acc_input_grad.ptr, acc_input_grad.rect, "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax backward time = %.2fms\n", elapsed);
  }
}

__host__
void Softmax::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  int idx = 0;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Softmax::measure_compute_time(Simulator* sim,
                                   const ParallelConfig& pc,
                                   float& forward_time,
                                   float& backward_time)
{
  //TODO: implement measure_forward
  return false;
}
