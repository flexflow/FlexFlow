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

Tensor FFModel::binary(OperatorType op,
                       const Tensor& in1,
                       const Tensor& in2,
                       bool inplace_a,
                       char const *name)
{
  ElementBinary *ele = new ElementBinary(*this, op, in1, in2, inplace_a, name);
  layers.push_back(ele);
  return ele->outputs[0];
}

Tensor FFModel::add(const Tensor& in1,
                    const Tensor& in2,
                    bool inplace_a,
                    char const *name)
{
  return this->binary(OP_EW_ADD, in1, in2, inplace_a, name);
}

Tensor FFModel::subtract(const Tensor& in1,
                         const Tensor& in2,
                         bool inplace_a,
                         char const *name)
{
  return this->binary(OP_EW_SUB, in1, in2, inplace_a, name);
}

Tensor FFModel::multiply(const Tensor& in1,
                         const Tensor& in2,
                         bool inplace_a,
                         char const *name)
{
  return this->binary(OP_EW_MUL, in1, in2, inplace_a, name);
}

Tensor FFModel::divide(const Tensor& in1,
                       const Tensor& in2,
                       bool inplace_a,
                       char const *name)
{
  return this->binary(OP_EW_DIV, in1, in2, inplace_a, name);
}

ElementBinary::ElementBinary(FFModel& model,
                             OperatorType _op_type,
                             const Tensor& in1,
                             const Tensor& in2,
                             bool _inplace_a,
                             const char* name)
: Op(
    model,
    _op_type,
    name,
    in1,
    in2
  ),
  inplace_a(_inplace_a)
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

bool ElementBinary::can_inplace_output(void)
{
  if (op_type == OP_EW_ADD)
    return true;
  if (op_type == OP_EW_MUL)
    return true;
  return false;
}

bool ElementBinary::has_inplace_output(void)
{
  return inplace_a;
}

void ElementBinary::do_inplace_output(void)
{
  inplace_a = true;
}

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
  if (inplace_a) {
    outputs[0] = inputs[0];
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
    for (int i = 0; i < 2; i++) {
      Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i].part.get_index_partition());
      // inplace_a require part_rect == inputs[0].part_rect
      if (i == 0)
        assert(input_rect == part_rect);
      if (input_rect == part_rect) {
        input_lps[i] = inputs[i].part;
        input_grad_lps[i] = inputs[i].part_grad;
      } else {
        model.create_disjoint_partition<NDIM>(
            inputs[i], IndexSpaceT<NDIM>(task_is), input_lps[i], input_grad_lps[i]);
      }
    }
    return;
  }
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  for (int i = 0; i < 2; i++) {
    Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
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
  ElementBinary* eb = (ElementBinary*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  ElementBinaryMeta* m = new ElementBinaryMeta(handle);
  m->op_type = eb->op_type;
  m->profiling = eb->profiling;
  m->inplace_a = eb->inplace_a;
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain;
  if (m->inplace_a) {
    assert(regions.size() == 2);
    assert(task->regions.size() == regions.size());
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    assert(output_domain == input_domain);
  } else {
    assert(regions.size() == 3);
    assert(task->regions.size() == regions.size());
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    assert(output_domain == input_domain);
  }
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
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  if (!inplace_a) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
    launcher.add_field(2, FID_DATA);
  } else {
    assert(outputs[0].part == input_lps[0]);
    assert(outputs[0].region == inputs[0].region);
  }
  //launcher.add_region_requirement(
  //  RegionRequirement(input_grad_lps[0], 0/*projection id*/,
  //    WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  //launcher.add_field(3, FID_DATA);
  //if (inputs[0].region_grad != inputs[1].region_grad) {
    // regions[4](I/O): input1_grad
  //  launcher.add_region_requirement(
  //    RegionRequirement(input_grad_lps[1], 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[1].region_grad));
  //  launcher.add_field(4, FID_DATA);
  //}
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

/*static*/
void ElementBinary::forward_kernel(const ElementBinaryMeta* m,
                                   const float* in1_ptr,
                                   const float* in2_ptr,
                                   float* out_ptr,
                                   cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

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
  //const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  Domain in1_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in2_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in1_domain == in2_domain);
  const float* in1_ptr = NULL, *in2_ptr = NULL;
  float *out_ptr = NULL;
  if (m->inplace_a) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    out_ptr = helperGetTensorPointerRW<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    in2_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    in1_ptr = out_ptr;
  } else {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    assert(out_domain == in1_domain);
    in1_ptr = helperGetTensorPointerRO<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    in2_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    out_ptr = helperGetTensorPointerWO<float>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  //print_tensor<float>(in1_ptr, in1_domain.get_volume(), "input1:");
  //print_tensor<float>(in2_ptr, in2_domain.get_volume(), "input2:");
  forward_kernel(m, in1_ptr, in2_ptr, out_ptr, stream);
  //print_tensor<float>(out_ptr, in1_domain.get_volume(), "output:");
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    char const *opName;
    switch (m->op_type) {
      case OP_EW_ADD:
        opName = "Add";
        break;
      case OP_EW_SUB:
        opName = "Sub";
        break;
      case OP_EW_MUL:
        opName = "Mul";
        break;
      case OP_EW_DIV:
        opName = "Div";
        break;
      default:
        assert(false);
    }
    printf("[%s] forward time (CF) = %.2fms\n", opName, elapsed);
  }
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
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  if (inplace_a) {
    assert(outputs[0].part == input_lps[0]);
    assert(outputs[0].region == inputs[0].region);
    launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[0].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(input_lps[1], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[1].region));
    launcher.add_field(1, FID_DATA);
  } else {
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
  }
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

/*static*/
void ElementBinary::backward_kernel(const ElementBinaryMeta* m,
                                    const float* out_grad_ptr,
                                    const float* in1_ptr,
                                    const float* in2_ptr,
                                    float* in1_grad_ptr,
                                    float* in2_grad_ptr,
                                    cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
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
}

/*
  regions[0](I or I/O): out_grad (I/O if inplace_a)
  regions[1](I): in0
  regions[2](I/O): in0_grad (Missing if in0_grad = out_grad)
  regions[3](I): in1 (Missing if in0 = in1)
  regions[4](I/O): in1_grad (Missing if in0=in1)
*/
void ElementBinary::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
  //const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  const float *in0_ptr = NULL, *in1_ptr = NULL, *out_grad_ptr = NULL;
  float *in0_grad_ptr = NULL, *in1_grad_ptr = NULL;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  if (m->inplace_a) {
    in0_grad_ptr = helperGetTensorPointerRW<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    if (regions.size() == 2 || regions.size() == 4);
    assert(task->regions.size() == regions.size());
    if (regions.size() == 2) {
      Domain in0_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
      out_grad_ptr = in0_grad_ptr;
    } else {
      Domain in0_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
      Domain in1_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      assert(in1_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = helperGetTensorPointerRO<float>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
      out_grad_ptr = in0_grad_ptr;
    }
  } else {
    assert(regions.size() == 3 || regions.size() == 5);
    assert(task->regions.size() == regions.size());
    out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    Domain in0_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    Domain in0_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
    assert(out_grad_domain == in0_grad_domain);
    assert(out_grad_domain == in0_domain);
    in0_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    in0_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
    if (regions.size() == 3) {
      // in0 == in1
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
    } else {
      Domain in1_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
      Domain in1_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[4].region.get_index_space());
      assert(out_grad_domain == in1_domain);
      assert(out_grad_domain == in1_grad_domain);
      in1_ptr = helperGetTensorPointerRO<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
        regions[4], task->regions[4], FID_DATA, ctx, runtime);
    }
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, out_grad_ptr, in0_ptr, in1_ptr, in0_grad_ptr, in1_grad_ptr, stream);
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
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  if (inplace_a) {
    // regions[0](I/O): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1](I): input0
    launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
    launcher.add_field(1, FID_DATA);
    if (inputs[0].region == inputs[1].region) {
      // regions[3](I): input1
      launcher.add_region_requirement(
        RegionRequirement(input_lps[1], 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[1].region));
      launcher.add_field(2, FID_DATA);
      // regions[4](I/O): input1_grad
      launcher.add_region_requirement(
        RegionRequirement(input_grad_lps[1], 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[1].region_grad));
      launcher.add_field(3, FID_DATA);
    }
  } else {
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
    // regions[2](I/O): input0_grad
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
    launcher.add_field(2, FID_DATA);
    if (inputs[0].region == inputs[1].region) {
      // regions[3](I): input1
      launcher.add_region_requirement(
        RegionRequirement(input_lps[1], 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[1].region));
      launcher.add_field(3, FID_DATA);
      // regions[4](I/O): input1_grad
      launcher.add_region_requirement(
        RegionRequirement(input_grad_lps[1], 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[1].region_grad));
      launcher.add_field(4, FID_DATA);
    }
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

bool ElementBinary::measure_operator_cost(Simulator* sim,
                                          const ParallelConfig& pc,
                                          CostMetrics& cost_metrics)
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
  Domain input_domain = sub_input0.get_domain();
  Domain output_domain = sub_output.get_domain();
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  sim->free_all();
  float* input0_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_ptr != NULL);
  float* input1_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  float* output_ptr = NULL;
  if (inplace_a) {
    output_ptr = input0_ptr;
  } else {
    output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  }
  assert(output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input0_ptr, input1_ptr, output_ptr, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input0_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    assert(input0_grad_ptr != NULL);
    float* input1_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    assert(input1_grad_ptr != NULL);
    float* output_grad_ptr = NULL;
    if (inplace_a) {
      output_grad_ptr = input0_grad_ptr;
    } else {
      output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    }
    assert(output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, output_grad_ptr, input0_ptr, input1_ptr, input0_grad_ptr, input1_grad_ptr, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Elewise Binary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Elewise Binary] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time);
  }

  return true;
}
