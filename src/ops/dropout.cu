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

Tensor FFModel::dropout(const Tensor& input,
                        float rate,
                        unsigned long long seed)
{
  // see = 0 is preserved as None, so we use a random seed
  if (seed == 0) {
    seed = std::rand();
  }
  Dropout *dropout = new Dropout(*this, input, rate, seed);
  layers.push_back(dropout);
  return dropout->outputs[0];
}

Dropout::Dropout(FFModel& model,
                 const Tensor& _input,
                 float _rate,
                 unsigned long long _seed)
: Op(model, OP_DROPOUT, "Dropout", _input), rate(_rate), seed(_seed)
{
  // Set output shape
  outputs[0].numDim = inputs[0].numDim;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = inputs[0].adim[i];
}

Tensor Dropout::init_inout(FFModel& model,
                           const Tensor& input)
{
  inputs[0] = input;
  create_output_and_partition(model);
  return outputs[0];
}

void Dropout::create_weights(FFModel& model)
{
  // Do nothing
}

void Dropout::create_output_and_partition(FFModel& model)
{
  int dim = inputs[0].numDim;
  switch (dim) {
    case 1:
    {
      task_is = model.get_or_create_task_is(1, name);
      create_output_and_partition_with_dim<1>(model);
      break;
    }
    case 2:
    {
      task_is = model.get_or_create_task_is(2, name);
      create_output_and_partition_with_dim<2>(model);
      break;
    }
    case 3:
    {
      task_is = model.get_or_create_task_is(3, name);
      create_output_and_partition_with_dim<3>(model);
      break;
    }
    case 4:
    {
      task_is = model.get_or_create_task_is(4, name);
      create_output_and_partition_with_dim<4>(model);
      break;
    }
    default:
    {
      assert(false && "Unsupported dim");
    }
  }
}

template<int NDIM>
void Dropout::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, IndexSpaceT<NDIM>(task_is), DT_FLOAT);
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

OpMeta* Dropout::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Dropout* dropout = (Dropout*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  DropoutMeta* m = new DropoutMeta(handle);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(input_domain == output_domain);
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreateDropoutDescriptor(&m->dropoutDesc));
  
  checkCUDNN(cudnnDropoutGetStatesSize(handle.dnn, &(m->dropoutStateSize)));
  checkCUDA(cudaMalloc(&m->dropoutStates, m->dropoutStateSize));
  checkCUDNN(cudnnSetDropoutDescriptor(m->dropoutDesc, handle.dnn,
      dropout->rate, m->dropoutStates, m->dropoutStateSize, dropout->seed));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  checkCUDNN(cudnnDropoutGetReserveSpaceSize(m->outputTensor, &(m->reserveSpaceSize)));
  checkCUDA(cudaMalloc(&m->reserveSpace, m->reserveSpaceSize));
  return m;
}

void Dropout::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
    case 1:
    {
      Rect<1> rect = domain;
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++) {
        FFHandler handle = ff.handlers[idx++];
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
      }
      break;
    }
    case 2:
    {
      Rect<2> rect = domain;
      int idx = 0;
      for (PointInRectIterator<2> it(rect); it(); it++) {
        FFHandler handle = ff.handlers[idx++];
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
      }
      break;
    }
    case 3:
    {
      Rect<3> rect = domain;
      int idx = 0;
      for (PointInRectIterator<3> it(rect); it(); it++) {
        FFHandler handle = ff.handlers[idx++];
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
      }
      break;
    }
    case 4:
    {
      Rect<4> rect = domain;
      int idx = 0;
      for (PointInRectIterator<4> it(rect); it(); it++) {
        FFHandler handle = ff.handlers[idx++];
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
      }
      break;
    }
    default:
      assert(false);
  }
  IndexLauncher init_launcher(DROPOUT_INIT_TASK_ID, task_is,
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
    case 1:
    {
      Rect<1> rect = domain;
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++) {
        meta[idx++] = fm.get_result<OpMeta*>(*it);
      }
      break;
    }
    case 2:
    {
      Rect<2> rect = domain;
      int idx = 0;
      for (PointInRectIterator<2> it(rect); it(); it++) {
        meta[idx++] = fm.get_result<OpMeta*>(*it);
      }
      break;
    }
    case 3:
    {
      Rect<3> rect = domain;
      int idx = 0;
      for (PointInRectIterator<3> it(rect); it(); it++) {
        meta[idx++] = fm.get_result<OpMeta*>(*it);
      }
      break;
    }
    case 4:
    {
      Rect<4> rect = domain;
      int idx = 0;
      for (PointInRectIterator<4> it(rect); it(); it++) {
        meta[idx++] = fm.get_result<OpMeta*>(*it);
      }
      break;
    }
  }
}

__host__
void Dropout::forward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  //float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* dropout = (const Dropout*) task->args;
  const DropoutMeta* m = *((DropoutMeta**) task->local_args);
  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDNN(cudnnDropoutForward(m->handle.dnn, m->dropoutDesc,
      m->inputTensor, input_ptr, m->outputTensor, output_ptr,
      m->reserveSpace, m->reserveSpaceSize));
}

void Dropout::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
    case 1:
    {
      Rect<1> rect = domain;
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 2:
    {
      Rect<2> rect = domain;
      int idx = 0;
      for (PointInRectIterator<2> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 3:
    {
      Rect<3> rect = domain;
      int idx = 0;
      for (PointInRectIterator<3> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 4:
    {
      Rect<4> rect = domain;
      int idx = 0;
      for (PointInRectIterator<4> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    default:
      assert(false);
  }
  IndexLauncher launcher(DROPOUT_FWD_TASK_ID, task_is,
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

/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/
__host__
void Dropout::backward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  //float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* dropout = (const Dropout*) task->args;
  const DropoutMeta* m = *((DropoutMeta**) task->local_args);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  checkCUDNN(cudnnDropoutBackward(m->handle.dnn, m->dropoutDesc,
      m->outputTensor, output_grad_ptr, m->inputTensor, input_grad_ptr,
      m->reserveSpace, m->reserveSpaceSize));
}

void Dropout::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
    case 1:
    {
      Rect<1> rect = domain;
      int idx = 0;
      for (PointInRectIterator<1> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 2:
    {
      Rect<2> rect = domain;
      int idx = 0;
      for (PointInRectIterator<2> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 3:
    {
      Rect<3> rect = domain;
      int idx = 0;
      for (PointInRectIterator<3> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    case 4:
    {
      Rect<4> rect = domain;
      int idx = 0;
      for (PointInRectIterator<4> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
      }
      break;
    }
    default:
      assert(false);
  }
  IndexLauncher launcher(DROPOUT_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementUnary)), argmap,
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

DropoutMeta::DropoutMeta(FFHandler handler)
: OpMeta(handler)
{}

bool Dropout::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  return false;
}
