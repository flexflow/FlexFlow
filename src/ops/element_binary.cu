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

Tensor FFModel::add(const Tensor& in1,
                    const Tensor& in2)
{
  ElementBinary *ele = new ElementBinary(*this, OP_EW_ADD, in1, in2);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementBinary* FFModel::add()
{
  ElementBinary* ele = new ElementBinary(*this, OP_EW_ADD);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::subtract(const Tensor& in1,
                         const Tensor& in2)
{
  ElementBinary *ele = new ElementBinary(*this, OP_EW_SUB, in1, in2);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementBinary* FFModel::subtract()
{
  ElementBinary* ele = new ElementBinary(*this, OP_EW_SUB);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::multiply(const Tensor& in1,
                         const Tensor& in2)
{
  ElementBinary *ele = new ElementBinary(*this, OP_EW_MUL, in1, in2);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementBinary* FFModel::multiply()
{
  ElementBinary* ele = new ElementBinary(*this, OP_EW_MUL);
  layers.push_back(ele);
  return ele;
}

Tensor FFModel::divide(const Tensor& in1,
                       const Tensor& in2)
{
  ElementBinary *ele = new ElementBinary(*this, OP_EW_DIV, in1, in2);
  layers.push_back(ele);
  return ele->outputs[0];
}

ElementBinary* FFModel::divide()
{
  ElementBinary* ele = new ElementBinary(*this, OP_EW_DIV);
  layers.push_back(ele);
  return ele;
}

ElementBinary::ElementBinary(FFModel& model,
                             OperatorType _op_type,
                             const Tensor& in1,
                             const Tensor& in2)
: Op(model, _op_type, "ElementBinary_"+std::to_string(_op_type), in1, in2), op_type(_op_type)
{
  //TODO: implement broadcast op
  numOutputs = 1;
  numWeights = 0;
  assert(in1.numDim == in2.numDim);
  int dim = in1.numDim;
  outputs[0].numDim = in1.numDim;
  for (int i = 0; i < dim; i++) {
    assert(in1.adim[i] == in2.adim[i]);
    outputs[0].adim[i] = in1.adim[i];
  }
}

ElementBinary::ElementBinary(FFModel& model,
                             OperatorType _op_type)
: Op(model, _op_type, "ElementBinary_"+std::to_string(_op_type), 2), op_type(_op_type)
{
}

Tensor ElementBinary::init_inout(FFModel& model,
                           const Tensor& input)
{
  // TODO: currently disable this functional API since
  // FlexFlow assumes a single tensor as input
  assert(false);
  Tensor in1 = input, in2 = input;
  inputs[0] = in1;
  inputs[1] = in2;
  create_output_and_partition(model);
  return outputs[0];
}

/*
void ElementBinary::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}
*/

void ElementBinary::create_weights(FFModel& model)
{
  // Do nothing
}

void ElementBinary::create_output_and_partition(FFModel& model)
{
  //TODO: implement broadcast op
  assert(inputs[0].numDim == inputs[1].numDim);
  int dim = inputs[0].numDim;
  for (int i = 0; i < dim; i++)
    assert(inputs[0].adim[i] == inputs[1].adim[i]);
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      create_output_and_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim for ElementWiseBinary operator
      assert(false);
    }
  }
}

template<int NDIM>
void ElementBinary::create_output_and_partition_with_dim(FFModel& model)
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
  for (int i = 0; i < 2; i++) {
    input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i].part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i].part;
      input_grad_lps[i] = inputs[i].part_grad;
    } else {
      model.create_disjoint_partition<NDIM>(
          inputs[i], IndexSpaceT<NDIM>(task_is), input_lps[i], input_grad_lps[i]);
    }
  }
}

__host__
OpMeta* ElementBinary::init_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  ElementBinary* eb = (ElementBinary*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  ElementBinaryMeta* m = new ElementBinaryMeta(handle);
  m->op_type = eb->op_type;
  cudnnOpTensorOp_t mode;
  switch (eb->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc, mode,
      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  return m;
}

void ElementBinary::init(const FFModel& ff)
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
  IndexLauncher launcher(ELEMENTBINARY_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementBinary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
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
void elewise_binary_forward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   OperatorType type,
                                   const float* in1,
                                   const float* in2,
                                   float* out)
{
    switch (type) {
      case OP_EW_ADD:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
	}
        break;
      }
      case OP_EW_SUB:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
        }
        break;
      }
      case OP_EW_MUL:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * in1[i] * in2[i] + beta * out[i];
        }
        break;
      }
      case OP_EW_DIV:
      {
        CUDA_KERNEL_LOOP(i, volume)
        {
          out[i] = alpha * (in1[i] / in2[i]) + beta * out[i];
        }
        break;
      }
      default:
        assert(false);
    }
}

void ElementBinary::forward_kernel(const ElementBinaryMeta* m,
                                   const float* in1_ptr,
                                   const float* in2_ptr,
                                   float* out_ptr) const
{
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (m->op_type) {
    case OP_EW_SUB:
      alpha2 = -1.0f;
      break;
    case OP_EW_ADD:
    case OP_EW_MUL:
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->inputTensor, in1_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->outputTensor, out_ptr));
}

/*
  regions[0](I): in1
  regions[1](I): in2
  regions[2](O): output
*/
__host__
void ElementBinary::forward_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  Domain in1_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in2_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  assert(in1_domain == in2_domain);
  assert(out_domain == in1_domain);

  const float* in1_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* in2_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  ele->forward_kernel(m, in1_ptr, in2_ptr, out_ptr);
  //elewise_binary_forward_kernel<<<GET_BLOCKS(out_domain.get_volume()), CUDA_NUM_THREADS>>>(
  //  out_domain.get_volume(), alpha, beta, ele->op_type, in1_ptr, in2_ptr, out_ptr);
}

void ElementBinary::forward(const FFModel& ff)
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
  IndexLauncher launcher(ELEMENTBINARY_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementBinary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

__global__
void elewise_binary_backward_kernel(coord_t volume,
                                    const float alpha,
                                    const float beta,
                                    OperatorType type,
                                    const float* out_grad,
                                    const float* in1,
                                    const float* in2,
                                    float* in1_grad,
                                    float* in2_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EW_ADD:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_SUB:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_MUL:
      {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_DIV:
      {
        in1_grad[i] = alpha * out_grad[i] / in2[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] * in1[i] / (in2[i] * in2[i]) + beta * in2_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

void ElementBinary::backward_kernel(const ElementBinaryMeta* m,
                                    const float* out_grad_ptr,
                                    const float* in1_ptr,
                                    const float* in2_ptr,
                                    float* in1_grad_ptr, 
                                    float* in2_grad_ptr) const
{
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
  switch (m->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->inputTensor, in1_grad_ptr));
  switch (m->op_type) {
    case OP_EW_ADD:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_SUB:
      alpha1 = -1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in1_ptr,
      &beta, m->inputTensor, in2_grad_ptr));
}

/*
  regions[0](I): out_grad
  regions[1](I): in0
  regions[2](I): in1
  regions[3](I/O): in0_grad
  regions[4](I/O): in1_grad (Missing if in0=in1)
*/
void ElementBinary::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
  const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  assert(regions.size() == 5 || regions.size() == 4);
  assert(task->regions.size() == regions.size());
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in0_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain in1_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  Domain in0_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  assert(out_grad_domain == in0_domain);
  assert(out_grad_domain == in1_domain);
  assert(out_grad_domain == in0_grad_domain);

  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* in1_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  const float* in2_ptr = helperGetTensorPointerRO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  float* in1_grad_ptr = helperGetTensorPointerRW<float>(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
  float* in2_grad_ptr = NULL;
  if (regions.size() == 5) {
    Domain in1_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
    assert(out_grad_domain == in1_grad_domain);
    in2_grad_ptr = helperGetTensorPointerRW<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  } else {
    in2_grad_ptr = in1_grad_ptr;
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  ele->backward_kernel(m, out_grad_ptr, in1_ptr, in2_ptr, in1_grad_ptr, in2_grad_ptr);
  //elewise_binary_backward_kernel<<<GET_BLOCKS(out_grad_domain.get_volume()), CUDA_NUM_THREADS>>>(
    //out_grad_domain.get_volume(), alpha, alpha, ele->op_type, out_grad_ptr, in1_ptr, in2_ptr,
    //in1_grad_ptr, in2_grad_ptr);
}

void ElementBinary::backward(const FFModel& ff)
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

  IndexLauncher launcher(ELEMENTBINARY_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementBinary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): input0
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): input1
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): input0_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(3, FID_DATA);
  if (inputs[0].region_grad != inputs[1].region_grad) {
    // regions[4](I/O): input1_grad
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[1], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[1].region_grad));
    launcher.add_field(4, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

ElementBinaryMeta::ElementBinaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  op_type = OP_ANY;
}

bool ElementBinary::measure_compute_time(Simulator* sim,
                                         const ParallelConfig& pc,
                                         float& forward_time,
                                         float& backward_time)
{
  Tensor sub_output, sub_input1, sub_input0;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input0, op_type))
    return false;
  if (!inputs[1].get_input_sub_tensor(pc, sub_input1, op_type))
    return false;
  ElementBinaryMeta* m = sim->ele_binary_meta;
  m->op_type = op_type;
  cudnnOpTensorOp_t mode;
  switch (op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc, mode,
      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  Domain input_domain, output_domain;
  input_domain.dim = sub_input0.numDim;
  for (int i = 0; i < sub_input0.numDim; i++) {
    input_domain.rect_data[i] = 0;
    input_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_input0.adim[i]-1;
  }
  output_domain.dim = sub_output.numDim;
  for (int i = 0; i < sub_output.numDim; i++) {
    output_domain.rect_data[i] = 0;
    output_domain.rect_data[i+Domain::MAX_RECT_DIM] = sub_output.adim[i]-1;
  }
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  sim->free_all();
  float* input0_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_ptr != NULL);
  float* input0_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_grad_ptr != NULL);
  float* input1_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  float* input1_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input1_grad_ptr != NULL);
  float* output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  // measure forward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    forward_kernel(m, input0_ptr, input1_ptr, output_ptr);
    //elewise_binary_forward_kernel<<<GET_BLOCKS(sub_output.get_volume()), CUDA_NUM_THREADS>>>(
    //    sub_output.get_volume(), alpha, beta, op_type,
    //    input0_ptr, input1_ptr, output_ptr);
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
    backward_kernel(m, output_ptr, input0_ptr, input1_ptr, input0_grad_ptr, input1_grad_ptr);
    //elewise_binary_backward_kernel<<<GET_BLOCKS(sub_output.get_volume()), CUDA_NUM_THREADS>>>(
    //    sub_output.get_volume(), alpha, alpha, op_type,
    //    output_ptr, input0_ptr, input1_ptr, input0_grad_ptr, input1_grad_ptr);
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  backward_time = milliseconds / sim->repeat_times;

  printf("[Measure Elewise Binary] num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
         sub_output.get_volume(), forward_time, backward_time);
  return true;
}
