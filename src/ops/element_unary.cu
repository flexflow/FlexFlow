/* Copyright 2020 Stanford
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

Tensor FFModel::exp(const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, OP_EXP, x);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementUnary* FFModel::exp()
{
  ElementUnary* ele = new ElementUnary(*this, OP_EXP);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::relu(const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, OP_RELU, x);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementUnary* FFModel::relu()
{
  ElementUnary* ele = new ElementUnary(*this, OP_RELU);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::sigmoid(const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, OP_SIGMOID, x);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementUnary* FFModel::sigmoid()
{
  ElementUnary* ele = new ElementUnary(*this, OP_SIGMOID);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::tanh(const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, OP_TANH, x);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementUnary* FFModel::tanh()
{
  ElementUnary* ele = new ElementUnary(*this, OP_TANH);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::elu(const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, OP_ELU, x);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementUnary* FFModel::elu()
{
  ElementUnary* ele = new ElementUnary(*this, OP_ELU);
  layers.push_back(ele);
  return ele;
}

ElementUnary::ElementUnary(FFModel& model,
                           OperatorType _op_type,
                           const Tensor& x)
: Op(model, _op_type, "ElementUnary_"+std::to_string(_op_type), x)
{
  outputs[0].numDim = inputs[0].numDim;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = inputs[0].adim[i];
}

ElementUnary::ElementUnary(FFModel& model,
                           OperatorType _op_type)
: Op(model, _op_type, "ElementUnary_"+std::to_string(_op_type), 1)
{}

Tensor ElementUnary::init_inout(FFModel& model,
                                const Tensor& input)
{
  inputs[0] = input;
  create_output_and_partition(model);
  return outputs[0];
}

bool ElementUnary::use_cudnn() const
{
  if (op_type == OP_RELU)
    return true;
  if (op_type == OP_SIGMOID)
    return true;
  if (op_type == OP_TANH)
    return true;
  if (op_type == OP_ELU)
    return true;
  return false;
}

/*
void ElementUnary::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}
*/

void ElementUnary::create_weights(FFModel& model)
{
  // Do nothing
}

void ElementUnary::create_output_and_partition(FFModel& model)
{
  int dim = inputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      task_is = model.get_or_create_task_is(DIM, name); \
      create_output_and_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim for ElementWiseUnary operator
      assert(false);
    }
  }
}

template<int NDIM>
void ElementUnary::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  Rect<NDIM> input_rect;
  input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
}

OpMeta* ElementUnary::init_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  ElementUnary* eu = (ElementUnary*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  ElementUnaryMeta* m = new ElementUnaryMeta(handle);
  if (eu->use_cudnn())
  {
    cudnnActivationMode_t mode;
    switch (eu->op_type) {
      case OP_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OP_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OP_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OP_ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    Domain input_domain = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());
    Domain output_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());

    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  }
  return m;
}

void ElementUnary::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      ParallelConfig pc; \
      std::string pcname = name; \
      ff.config.find_parallel_config(DIM, pcname, pc); \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        FFHandler handle = ff.handlers[pc.device_ids[idx++]]; \
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher init_launcher(ELEMENTUNARY_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(ElementUnary)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              FFConfig::get_hash_id(std::string(name)));
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

__global__
void elewise_unary_forward_kernel(coord_t volume,
                                  const float alpha,
                                  const float beta,
                                  OperatorType type,
                                  const float* in,
                                  float* out)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        out[i] = alpha * exp(in[i]) + beta * out[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
__host__
void ElementUnary::forward_task(const Task* task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime* runtime)
{
  float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_domain == input_domain);

  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  if (ele->use_cudnn()) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->inputTensor, input_ptr,
        &beta, m->outputTensor, output_ptr));
  } else {
    elewise_unary_forward_kernel<<<GET_BLOCKS(output_domain.get_volume()), CUDA_NUM_THREADS>>>(
    output_domain.get_volume(), alpha, beta, ele->op_type, input_ptr, output_ptr);
  }
  
}

void ElementUnary::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(ELEMENTUNARY_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementUnary)), argmap,
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

__global__
void elewise_unary_backward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   OperatorType type,
                                   const float* output_grad,
                                   const float* input,
                                   float* input_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        //TODO: change to use output instead of recomputing
        input_grad[i] = alpha * output_grad[i] * exp(input[i]) + beta * input_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
__host__
void ElementUnary::backward_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  float alpha = 1.0f;
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  Domain output_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  assert(output_grad_domain == input_domain);
  assert(output_grad_domain == output_domain);
  assert(output_grad_domain == input_grad_domain);

  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  const float* output_ptr = helperGetTensorPointerRO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  if (ele->use_cudnn()) {
    checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc, 
        &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
        m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
  } else {
    elewise_unary_backward_kernel<<<GET_BLOCKS(input_domain.get_volume()), CUDA_NUM_THREADS>>>(
        input_domain.get_volume(), alpha, alpha, ele->op_type, output_grad_ptr, input_ptr, input_grad_ptr);
  }
}

void ElementUnary::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }

  IndexLauncher launcher(ELEMENTUNARY_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementUnary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): input
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

ElementUnaryMeta::ElementUnaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

bool ElementUnary::measure_compute_time(Simulator* sim,
                                        const ParallelConfig& pc,
                                        float& forward_time,
                                        float& backward_time)
{
  Tensor sub_output, sub_input;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, op_type))
    return false;
  ElementUnaryMeta* m = sim->ele_unary_meta;
  if (use_cudnn())
  {
    cudnnActivationMode_t mode;
    switch (op_type) {
      case OP_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OP_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OP_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OP_ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    Domain input_domain, output_domain;
    input_domain.dim = sub_input.numDim;
    for (int i = 0; i < sub_input.numDim; i++) {
      input_domain.rect_data[i] = 0;
      input_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_input.adim[i]-1;
    }
    output_domain.dim = sub_output.numDim;
    for (int i = 0; i < sub_output.numDim; i++) {
      output_domain.rect_data[i] = 0;
      output_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_output.adim[i]-1;
    }
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  }
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_grad_ptr != NULL);
  float* output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_grad_ptr != NULL);

  float alpha = 1.0f, beta = 0.0f;
  // measure forward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    if (use_cudnn()) {
      checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
          &alpha, m->inputTensor, input_ptr,
          &beta, m->outputTensor, output_ptr));
    } else {
      elewise_unary_forward_kernel<<<GET_BLOCKS(sub_output.get_volume()), CUDA_NUM_THREADS>>>(
          sub_output.get_volume(), alpha, beta, op_type,
          input_ptr, output_ptr);
    }
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  forward_time = milliseconds / sim->repeat_times;

  // measure backward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    if (use_cudnn()) {
      checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc, 
          &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
          m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
    } else {
      elewise_unary_backward_kernel<<<GET_BLOCKS(sub_output.get_volume()), CUDA_NUM_THREADS>>>(
          sub_output.get_volume(), alpha, alpha, op_type,
          output_ptr, input_ptr, input_grad_ptr);
    }
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  backward_time = milliseconds / sim->repeat_times;

  printf("[Measure Elewise Unary] num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
         sub_output.get_volume(), forward_time, backward_time);
  return true;
}
