/* Copyright 2020 Stanford, Facebook
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

// return A*B+C
BatchMatmul::BatchMatmul(FFModel& model,
                         const Tensor& A,
                         const Tensor& B)
: Op(model, OP_BATCHMATMUL, "BatchMatmul_", A, B)
{
  assert(A.numDim == B.numDim);
  for (int i = A.numDim-1; i >= 2; i++)
    assert(A.adim[i] == B.adim[i]);
  assert(A.adim[0] == B.adim[1]);
  outputs[0].numDim = A.numDim;
  for (int i = 0; i < A.numDim; i++)
    outputs[0].adim[i] = A.adim[i];
  outputs[0].adim[0] = B.adim[0];
  // C is not none
  //if (C != Tensor::NO_TENSOR) {
  //  numInputs = 3;
  //  assert(C.numDim == outputs[0].numDim);
  //  for (int i = 0; i < C.numDim; i++)
  //    assert(C.adim[i] == outputs[0].adim[i]);
  //} else {
    numInputs = 2;
  //}
  numWeights = 0;
}

Tensor BatchMatmul::init_inout(FFModel& model, const Tensor& _input)
{
  // Deprecated APIs --- remove soon
  assert(false);
  return outputs[0];
}

void BatchMatmul::create_weights(FFModel& model)
{
  // Do nothing since we don't have any weights
}

void BatchMatmul::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace
  int dim = inputs[0].numDim;
  assert(dim == inputs[1].numDim);
  switch (dim) {
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
      // Unsupported dim for BatchMatmul operator
      assert(false);
    }
  }
}

template<int NDIM>
void BatchMatmul::create_output_and_partition_with_dim(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // currently only support data parallel for batch matmul
  // the parallel degree of the inner most two dims must be 1
  assert(part_rect.hi[0] == part_rect.lo[0]);
  assert(part_rect.hi[1] == part_rect.lo[1]);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, (IndexSpaceT<NDIM>)task_is, DT_FLOAT);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  for (int i = 0; i < numInputs; i++) {
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
OpMeta* BatchMatmul::init_task(const Task* task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  FFHandler handle = *((const FFHandler*) task->local_args);
  BatchMatmulMeta* m = new BatchMatmulMeta(handle);
  return m;
}

void BatchMatmul::init(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
    case 3:
    {
      init_with_dim<3>(ff);
      break;
    }
    case 4:
    {
      init_with_dim<4>(ff);
      break;
    }
    default:
      assert(false);
  }
}

template<int NDIM>
void BatchMatmul::init_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchMatmul)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i+1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }

}

/*
A: (batch, n, k)
B: (batch, k, m)
O: (batch, n, m)
O = A * B
*/
void BatchMatmul::forward_kernel(const BatchMatmulMeta* meta,
                                 float* o_ptr,
                                 const float* a_ptr,
                                 const float* b_ptr,
                                 const float* c_ptr,
                                 int m, int n, int k, int batch) const
{
  int a_stride = n * k;
  int b_stride = m * k;
  int o_stride = n * m;
  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemmStridedBatched(meta->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k, &alpha, b_ptr, m, b_stride, a_ptr, k, a_stride,
      &beta, o_ptr, m, o_stride, batch));
  // current assume c is null
  assert(c_ptr == NULL);
}

/*
  regions[0](O): output
  regions[1](I): A
  regions[2](I): B
  (optional) regions[3](I): C
  output = A * B + C
*/
__host__
void BatchMatmul::forward_task(const Task* task,
                               const std::vector<PhysicalRegion>& regions,
                               Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const BatchMatmul* bmm = (const BatchMatmul*) task->args;
  const BatchMatmulMeta* meta = *((BatchMatmulMeta**) task->local_args);
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain a_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain b_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
  assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
  assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
  int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
  assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
  assert(a_domain.get_dim() == b_domain.get_dim());
  assert(a_domain.get_dim() == out_domain.get_dim());
  int batch = 1;
  for (int i = 2; i < a_domain.get_dim(); i++) {
    int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
    assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
    assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
    batch *= dim_size;
  }
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* a_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  const float* b_ptr = helperGetTensorPointerRO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  const float* c_ptr = NULL;
  if (regions.size() == 4) {
    Domain c_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
    assert(c_domain == a_domain);
    c_ptr = helperGetTensorPointerRO<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }
  cudaEvent_t t_start, t_end;
  if (bmm->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(meta->handle.blas, stream));
  checkCUDNN(cudnnSetStream(meta->handle.dnn, stream));
#endif
  bmm->forward_kernel(meta, out_ptr, a_ptr, b_ptr, c_ptr,
    m, n, k, batch);
  if (bmm->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchMatmul forward time = %.2lfms\n", elapsed);
  }
}

void BatchMatmul::forward(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
    case 3:
    {
      forward_with_dim<3>(ff);
      break;
    }
    case 4:
    {
      forward_with_dim<4>(ff);
      break;
    }
    default:
      assert(false);
  }
}

template<int NDIM>
void BatchMatmul::forward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHMATMUL_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchMatmul)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
A, AGrad: (batch, n, k)
B, BGrad: (batch, k, m)
O, OGrad: (batch, n, m)
AGrad = OGrad * B^T
BGrad = A^T * OGrad
*/
void BatchMatmul::backward_kernel(const BatchMatmulMeta* meta,
                                  const float* o_ptr,
                                  const float* o_grad_ptr,
                                  const float* a_ptr,
                                  float* a_grad_ptr,
                                  const float* b_ptr,
                                  float* b_grad_ptr,
                                  float* c_grad_ptr,
                                  int m, int n, int k, int batch) const
{
  int a_stride = n * k;
  int b_stride = m * k;
  int o_stride = n * m;
  float alpha = 1.0f;
  checkCUDA(cublasSgemmStridedBatched(meta->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
      k, n, m, &alpha, b_ptr, m, b_stride, o_grad_ptr, m, o_stride,
      &alpha, a_grad_ptr, k, a_stride, batch));
  checkCUDA(cublasSgemmStridedBatched(meta->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
      m, k, n, &alpha, o_grad_ptr, m, o_stride, a_ptr, k, a_stride,
      &alpha, b_grad_ptr, m, b_stride, batch));
}


/*
  regions[0](I): output
  regions[1](I): output_grad
  regions[2](I): A
  regions[3](I/O): A_grad
  regions[4](I): B
  regions[5](I/O): B_grad
  regions[6](I/O): C_grad
*/
__host__
void BatchMatmul::backward_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
  // Currently assume C is NULL
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  BatchMatmul* bmm = (BatchMatmul*) task->args;
  const BatchMatmulMeta* meta = *((BatchMatmulMeta**) task->local_args);
  // output domains
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(out_domain == out_grad_domain);
  // A domains
  Domain a_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  Domain a_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  assert(a_domain == a_grad_domain);
  // B domains
  Domain b_domain = runtime->get_index_space_domain(
    ctx, task->regions[4].region.get_index_space());
  Domain b_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[4].region.get_index_space());
  assert(b_domain == b_grad_domain);
  // check dins
  int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
  assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
  assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
  int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
  assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
  assert(a_domain.get_dim() == b_domain.get_dim());
  assert(a_domain.get_dim() == out_domain.get_dim());
  int batch = 1;
  for (int i = 2; i < a_domain.get_dim(); i++) {
    int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
    assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
    assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
    batch *= dim_size;
  }
  // get pointers
  const float* out_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  const float* a_ptr = helperGetTensorPointerRO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  float* a_grad_ptr = helperGetTensorPointerRW<float>(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
  const float* b_ptr = helperGetTensorPointerRO<float>(
    regions[4], task->regions[4], FID_DATA, ctx, runtime);
  float* b_grad_ptr = helperGetTensorPointerRW<float>(
    regions[5], task->regions[5], FID_DATA, ctx, runtime);

  float* c_grad_ptr = NULL;
  cudaEvent_t t_start, t_end;
  if (bmm->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(meta->handle.blas, stream));
  checkCUDNN(cudnnSetStream(meta->handle.dnn, stream));
#endif
  bmm->backward_kernel(meta, out_ptr, out_grad_ptr, a_ptr, a_grad_ptr,
    b_ptr, b_grad_ptr, c_grad_ptr, m, n, k, batch);
  if (bmm->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchMatmul forward time = %.2lfms\n", elapsed);
  }
}

void BatchMatmul::backward(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
    case 3:
    {
      backward_with_dim<3>(ff);
      break;
    }
    case 4:
    {
      backward_with_dim<4>(ff);
      break;
    }
    default:
      assert(false);
  }
}

/*
  regions[0](I): output
  regions[1](I): output_grad
  regions[2](I): A
  regions[3](I/O): A_grad
  regions[4](I): B
  regions[5](I/O): B_grad
  regions[6](I/O): C_grad
*/
template<int NDIM>
void BatchMatmul::backward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHMATMUL_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchMatmul)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): A
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): A_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): B
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): B_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(5, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

__host__
void BatchMatmul::print_layer(const FFModel& ff)
{
  return;
}

BatchMatmulMeta::BatchMatmulMeta(FFHandler handler)
: OpMeta(handler)
{}

bool BatchMatmul::measure_compute_time(Simulator* sim,
                                       const ParallelConfig& pc,
                                       float& forward_time,
                                       float& backward_time)
{
  // To be implemented
  assert(false);
  return false;
}
