/* Copyright 2020 Facebook
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

using namespace Legion;

Tensor FFModel::batch_matmul(const Tensor A,
                             const Tensor B,
                             int a_seq_length_dim,
                             int b_seq_length_dim)
{
  BatchMatmul* bmm = new BatchMatmul(*this, A, B,
      a_seq_length_dim, b_seq_length_dim);
  layers.push_back(bmm);
  return bmm->outputs[0];
}

// return A*B
BatchMatmul::BatchMatmul(FFModel& model,
                         const Tensor A,
                         const Tensor B,
                         int _a_seq_length_dim,
                         int _b_seq_length_dim)
: Op(model, OP_BATCHMATMUL, "BatchMatmul_", 2/*inputs*/, 0/*weights*/, A, B),
  a_seq_length_dim(A->num_dims-1-_a_seq_length_dim),
  b_seq_length_dim(B->num_dims-1-_b_seq_length_dim)
{
  assert((a_seq_length_dim <= 1) && "FlexFlow currently only supports seq_length_dim of 0 or 1 (in Fortran ordering).");
  assert((b_seq_length_dim <= 1) && "FlexFlow currently only supports seq_length_dim of 0 or 1 (in Fortran ordering).");
  assert(A->num_dims == B->num_dims);
  for (int i = A->num_dims-1; i >= 2; i--)
    assert(A->dims[i] == B->dims[i]);
  assert(A->dims[0] == B->dims[1]);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < A->num_dims; i++)
    dims[i] = A->dims[i];
  dims[0] = B->dims[0];
  numOutputs = 1;
  outputs[0] = model.create_tensor_legion_ordering(A->num_dims, dims, DT_FLOAT, this);
  // C is not none
  //if (C != Tensor::NO_TENSOR) {
  //  numInputs = 3;
  //  assert(C.num_dims == outputs[0].num_dims);
  //  for (int i = 0; i < C.num_dims; i++)
  //    assert(C.adim[i] == outputs[0].adim[i]);
  //}
}

#ifdef DEADCODE
void BatchMatmul::create_input_partition(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain part_rect = runtime->get_index_space_domain(ctx, task_is);
  // currently only support data parallel for batch matmul
  // the parallel degree of the inner most two dims must be 1
  assert(part_rect.hi()[0] == part_rect.lo()[0]);
  assert(part_rect.hi()[1] == part_rect.lo()[1]);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  for (int i = 0; i < numInputs; i++) {
    Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i]->part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i]->part;
      input_grad_lps[i] = inputs[i]->part_grad;
    } else {
      model.create_disjoint_partition<NDIM>(
          inputs[i], IndexSpaceT<NDIM>(task_is), input_lps[i], input_grad_lps[i]);
    }
  }
#endif
}
#endif

__host__
OpMeta* BatchMatmul::init_task(const Task* task,
                               const std::vector<PhysicalRegion>& regions,
                               Context ctx, Runtime* runtime)
{
  const BatchMatmul* bmm = (BatchMatmul*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  BatchMatmulMeta* m = new BatchMatmulMeta(handle);
  m->profiling = bmm->profiling;
  m->a_seq_length_dim = bmm->a_seq_length_dim;
  m->b_seq_length_dim = bmm->b_seq_length_dim;
  return m;
}

void BatchMatmul::init(const FFModel& ff)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      init_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void BatchMatmul::init_with_dim(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(BatchMatmul)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
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
                                 int m, int n, int k,
                                 int batch,
                                 int a_seq_length_dim,
                                 int b_seq_length_dim,
                                 int seq_length)
{
  //int a_stride = n * k;
  //int b_stride = m * k;
  //int o_stride = n * m;
  int lda = k; int ldb = m; int ldo = m;
  long long int strideA = (long long int)n*k;
  long long int strideB = (long long int)k*m;
  long long int strideO = (long long int)n*m;
  if ((a_seq_length_dim==0)&&(seq_length>=0)) {
    assert(seq_length <= k);
    k = seq_length;
    assert(b_seq_length_dim == 1);
  } else if ((a_seq_length_dim==1)&&(seq_length>=0)) {
    assert(seq_length <= n);
    n = seq_length;
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((a_seq_length_dim<0)||(seq_length<0));
  }
  if ((b_seq_length_dim==0)&&(seq_length>=0)) {
    assert(seq_length <= m);
    m = seq_length;
  } else if ((b_seq_length_dim==1)&&(seq_length>=0)) {
    assert(a_seq_length_dim == 0);
    assert(k == seq_length);
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((b_seq_length_dim<0)||(seq_length<0));
  }

  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemmStridedBatched(meta->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k, &alpha, b_ptr, ldb, strideB, a_ptr, lda, strideA,
      &beta, o_ptr, ldo, strideO, batch));
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
  //const BatchMatmul* bmm = (const BatchMatmul*) task->args;
  const FFIterationConfig* iter_config = (const FFIterationConfig*) task->args;
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
      ctx, task->regions[3].region.get_index_space());
    assert(c_domain == a_domain);
    c_ptr = helperGetTensorPointerRO<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }
  cudaEvent_t t_start, t_end;
  if (meta->profiling) {
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
  forward_kernel(meta, out_ptr, a_ptr, b_ptr, c_ptr,
    m, n, k, batch, meta->a_seq_length_dim, meta->b_seq_length_dim,
    iter_config->seq_length);
  if (meta->profiling) {
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
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      forward_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
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
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_FWD_TASK_ID, parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
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
                                  int m, int n, int k, int batch)
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
  assert (c_grad_ptr == NULL);
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
  //BatchMatmul* bmm = (BatchMatmul*) task->args;
  const FFIterationConfig* iter_config = (const FFIterationConfig*) task->args;
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
  if (meta->profiling) {
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
  // TODO: add support for meta->a_seq_length_dim >= 0
  // or meta->b_seq_length_dim >= 0
  assert((meta->a_seq_length_dim<0)||(iter_config->seq_length==0));
  assert((meta->b_seq_length_dim<0)||(iter_config->seq_length==0));
  backward_kernel(meta, out_ptr, out_grad_ptr, a_ptr, a_grad_ptr,
    b_ptr, b_grad_ptr, c_grad_ptr, m, n, k, batch);
  if (meta->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchMatmul backward time = %.2lfms\n", elapsed);
  }
}

void BatchMatmul::backward(const FFModel& ff)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      backward_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
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
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_BWD_TASK_ID, parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): A
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): A_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): B
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): B_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
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

bool BatchMatmul::measure_operator_cost(Simulator* sim,
                                        const ParallelConfig& pc,
                                        CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input0, sub_input1;
  if (! outputs[0]->get_output_sub_tensor(pc, sub_output, OP_BATCHMATMUL)) {
    return false;
  }
  if (! inputs[0]->get_input_sub_tensor(pc, sub_input0, OP_BATCHMATMUL)) {
    return false;
  }
  if (! inputs[1]->get_input_sub_tensor(pc, sub_input1, OP_BATCHMATMUL)) {
    return false;
  }

  int input0_c = sub_input0.dims[0].size;
  int input0_r = sub_input0.dims[1].size;
  int input1_c = sub_input1.dims[0].size;
  int input1_r = sub_input1.dims[1].size;
  int output_c = sub_output.dims[0].size;
  int output_r = sub_output.dims[1].size;

  assert (input0_c == input1_r);
  assert (input0_r == output_r);
  assert (input1_c == output_c);

  assert (sub_input0.dims[2] == sub_input1.dims[2]);
  assert (sub_input1.dims[2] == sub_output.dims[2]);
  int batch = 1;
  assert(sub_input0.num_dims == sub_input1.num_dims);
  for (int i = 2; i < sub_input0.num_dims; i++) {
    assert(sub_input0.dims[i] == sub_input1.dims[i]);
    assert(sub_input0.dims[i] == sub_output.dims[i]);
    batch *= sub_input0.dims[i].size;
  }

  BatchMatmulMeta *meta = sim->batch_matmul_meta;

  // allocate tensors in simulator
  sim->free_all();
  float *a_ptr = (float *)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert (a_ptr != NULL);
  float *b_ptr = (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert (b_ptr != NULL);
  float *c_ptr = NULL;
  float *out_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (out_ptr != NULL);

  int m = input1_c;
  int n = input0_r;
  int k = input0_c;

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(meta, out_ptr, a_ptr, b_ptr, c_ptr, m, n, k, batch);
  };

  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *a_grad_ptr = (float *)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    float *b_grad_ptr = (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
    float *c_grad_ptr = NULL;
    float *out_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (out_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(meta, out_ptr, out_grad_ptr, a_ptr, a_grad_ptr, b_ptr, b_grad_ptr, c_grad_ptr, m, n, k, batch);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure BatchMatmul] name(%s) adim(%d %d %d) bdim(%d %d %d) odim(%d %d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        batch, input0_r, input0_c,
        batch, input1_r, input1_c,
        batch, output_r, output_c,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure BatchMatmul] name(%s) adim(%d %d %d) bdim(%d %d %d) odim(%d %d %d) forward_time(%.4lf)\n",
        name,
        batch, input0_r, input0_c,
        batch, input1_r, input1_c,
        batch, output_r, output_c,
        cost_metrics.forward_time);
  }

  return true;
}
