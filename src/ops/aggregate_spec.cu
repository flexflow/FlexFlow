/* Copyright 2019 Stanford
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

#define MAX_K 4
#define MAX_BATCH_SIZE 32
#define MAX_N 12


Tensor FFModel::aggregate_spec(const Tensor* inputs, /* gate_preds, gate_assign, full_gate_pred, n * exp_pred */
                          int n, float lambda_bal, const char* name)
{
  AggregateSpec* aggr = new AggregateSpec(*this, inputs, n, lambda_bal, name);
  layers.push_back(aggr);
  return aggr->outputs[0];
}


AggregateSpec::AggregateSpec(FFModel& model,
                    const Tensor* _inputs,
                    int _n, float _lambda_bal, const char* name)
: Op(model, OP_AGG_SPEC, name, _n+4, _inputs),
  n(_n), lambda_bal(_lambda_bal),
  profiling(model.config.profiling)
{
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(n <= MAX_N && "Increase MAX_N in #define");
  assert(inputs[0].adim[0] <= MAX_K && "Increase MAX_K in #define");
  assert(inputs[0].adim[1] <= MAX_BATCH_SIZE && "Increase MAX_BATCH_SIZE in #define");

  assert(n+4 == numInputs);
  assert(n > 0);
  assert(inputs[0].numDim == 2);
  assert(inputs[1].numDim == 2);
  assert(inputs[2].numDim == 2);
  assert(inputs[3].numDim == 2);

  for(int i = 0; i < inputs[0].numDim; i++) {
    assert(inputs[0].adim[i] == inputs[1].adim[i]);
    assert(inputs[0].adim[i] == inputs[2].adim[i]);
  }
  assert(inputs[0].adim[1] == inputs[3].adim[1]);
  assert(inputs[3].adim[0] == n);

  // expert inputs
  int num_dim = inputs[4].numDim;
  int out_dim = inputs[4].adim[0];
  for(int i = 1; i < n; i++) {
    assert(inputs[i+4].numDim == num_dim);
    assert(inputs[i+4].adim[0] == out_dim);
  }
  // output
  outputs[0].numDim = num_dim;
  int k = inputs[0].adim[0];
  for(int i = 0; i < num_dim-1; i++)
    outputs[0].adim[i] = inputs[4].adim[i];
  outputs[0].adim[num_dim-1] = k*inputs[0].adim[num_dim-1];

  numWeights = 0;
}


void AggregateSpec::create_weights(FFModel& model)
{
  // Do nothing
}


void AggregateSpec::create_output_and_partition(FFModel& model)
{
  // Retrieve the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);

  // Can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);

  int batch_size = inputs[0].adim[1];
  int out_dim = inputs[4].adim[0];
  int k = inputs[0].adim[0];

  const int dims[2] = {k*batch_size, out_dim};
  outputs[0] = model.create_tensor<2>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;


  // Compute partition bound for input
  for(int i = 0; i < n+4; i++) {
    Rect<2> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i].part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i].part;
      input_grad_lps[i] = inputs[i].part_grad;
    } else {
      model.create_disjoint_partition<2>(
        inputs[i], (IndexSpaceT<2>)task_is, input_lps[i], input_grad_lps[i]);
    }
  }
}


OpMeta* AggregateSpec::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  AggregateSpec* agg = (AggregateSpec*) task->args;
  FFHandler handle = *((FFHandler*)task->local_args);
  AggregateSpecMeta* m = new AggregateSpecMeta(handle, agg->n);
  m->profiling = agg->profiling;
  return m;
}


void AggregateSpec::init(const FFModel& ff)
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
  IndexLauncher launcher(AGG_SPEC_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(AggregateSpec)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
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
void aggspec_forward_kernel(float** exp_preds,
        const int* exp_assign,
        float* output,
        int n, // num experts
        const int k, // num chosen experts
        int exp_samples, // max samples per expert
        const int batch_size,
        int out_dim)
{
  __shared__ float* chosen_exp_preds[MAX_K*MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if(threadIdx.x == 0) {
    int expert_idx[MAX_N] = {0};
    for(int i = 0; i < batch_size; i++) {
      for(int j = 0; j < k; j++) {
        // Get pointer to chosen expert predictions
        int expert = exp_assign[i*k+j];
        if(expert_idx[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_preds[i*k+j] = 0;
          continue;
        }
        chosen_exp_preds[i*k+j] = exp_preds[expert] + expert_idx[expert]*out_dim;
        expert_idx[expert]++;
      }
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*batch_size*out_dim)
  {
    if(chosen_exp_preds[i/out_dim] != 0) {
      output[i] = chosen_exp_preds[i/out_dim][i%out_dim];
    }
    else {
      output[i] = 0.0f;
    }
  }
}


__device__
void aggspec_backward_kernel_gate(const float* output_grad,
              float* full_gate_grads,
              const int* expert_assign,
              const bool* cache_corr,
              const float* gate_pred,
              int* expert_bal, float lambda_bal,
              int batch_size, int k, int n, int out_dim)
{

  __shared__ float gate_grad_sum[MAX_BATCH_SIZE];

  // init gate_grad_sum to 0
  CUDA_KERNEL_LOOP(i, batch_size)
  {
    gate_grad_sum[i] = 0.0f;
  }

  __syncthreads();

  // get sum of expert errors
  /* NOTE: Errors just squared L2 norm of gradients. * batch_size because the
  expert gradients are /= batch_size and then it would be /= batch_size^2 here */
  CUDA_KERNEL_LOOP(i, batch_size*k*out_dim)
  {
    if(cache_corr[i/(k*out_dim)]) {
      float res = output_grad[i] * output_grad[i] * batch_size;
      float* gate_grad_idx = full_gate_grads + (i/(out_dim*k))*n
        + expert_assign[(i/(out_dim*k))*k+(i/out_dim)%k];
      atomicAdd(gate_grad_idx, res);
      atomicAdd(gate_grad_sum+i/(k*out_dim), res);
    }
  }

  // Compute gate gradients:
  // Assigned expert i, sample j: pred(i,j) - err_(i,j)/sum_l err(l,j)
  __syncthreads();
  CUDA_KERNEL_LOOP(i, k*batch_size)
  {
    if(cache_corr[i/k]) {
      full_gate_grads[i/k*n + expert_assign[i]] /= gate_grad_sum[i/k];
      full_gate_grads[i/k*n + expert_assign[i]] -= (1.0f - gate_pred[i]);
    }
  }

  // balance term
  __syncthreads();
  CUDA_KERNEL_LOOP(i, n*batch_size)
  {
    full_gate_grads[i] += lambda_bal*expert_bal[i%n];
  }

  __syncthreads();

  // make 0 mean
  CUDA_KERNEL_LOOP(i, n*batch_size)
  {
    int start = (i/n)*n;
    float sub = -full_gate_grads[i]/n;
    for(int j = 0; j < n; j++) {
      atomicAdd(full_gate_grads+start+j, sub);
    }
  }
}


__device__
void aggspec_backward_kernel_exp(const float* output_grad,
              const float* gate_preds,
              float** exp_grads,
              int batch_size,
              int k,
              int out_dim) {
  // compute expert gradients
  CUDA_KERNEL_LOOP(i, k*out_dim*batch_size)
  {
    if (exp_grads[i/out_dim] != 0) {
      exp_grads[i/out_dim][i%out_dim] += gate_preds[i/out_dim] * output_grad[i];
    }
  }
}


__global__
void aggspec_backward_kernel(float** exp_grads,
        const int* exp_assign,
        const int* true_exp_assign,
        const float* gating_net_preds,
        float* full_gating_grads,
        const float* output_grads,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        float lambda_bal,
        int batch_size,
        int out_dim)
{
  __shared__ float* chosen_exp_grads[MAX_K*MAX_BATCH_SIZE];
  __shared__ int expert_bal[MAX_N];
  __shared__ bool cache_corr[MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if(threadIdx.x == 0) {
    // init arrays
    for(int i = 0; i < n; i++) expert_bal[i] = 0;
    for(int i = 0; i < batch_size; i++) cache_corr[i] = true;

    // Get pointer to chosen expert grads and expert counts
    for(int i = 0; i < batch_size; i++) {
      for(int j = 0; j < k; j++) {
        int expert = true_exp_assign[k*i + j];
        if(expert != exp_assign[k*i + j])
          cache_corr[i] = false;
        if(expert_bal[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_grads[i*k+j] = 0;
          expert_bal[expert]++;
          continue;
        }
        chosen_exp_grads[i*k+j] = exp_grads[expert] + expert_bal[expert]*out_dim;
        expert_bal[expert]++;
      }
    }
  }

  __syncthreads();

  // NOTE: These 2 functions could execute independently in parallel
  // get expert gradients
  aggspec_backward_kernel_exp(output_grads, gating_net_preds, chosen_exp_grads,
    batch_size, k, out_dim);

  // get gating net gradients
  aggspec_backward_kernel_gate(output_grads, full_gating_grads, exp_assign,
    cache_corr, gating_net_preds, expert_bal, (lambda_bal*n)/batch_size,
    batch_size, k, n, out_dim);
}


void AggregateSpec::forward_task(const Task *task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  int n = ((AggregateSpec*)task->args)->n;

  assert((int)regions.size() == n+3);
  assert((int)task->regions.size() == n+3);

  const AggregateSpecMeta* m = *((AggregateSpecMeta**)task->local_args);

  // get gate_pred, gate_assign, output
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[n+2], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[n+2].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  coord_t k = rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1;
  assert(k == rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);
  assert(k*batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  cudaStream_t stream;
  checkCUDA(create_stream(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  // call forward kernel
  cudaMemcpy(m->dev_region_ptrs, exp_preds, n*sizeof(float*), cudaMemcpyHostToDevice);

  aggspec_forward_kernel<<<GET_BLOCKS(batch_size*k*out_dim),
    min(CUDA_NUM_THREADS,(int)(batch_size*k*out_dim)), 0, stream>>>(m->dev_region_ptrs,
    acc_gate_assign.ptr(rect_gate_assign), acc_output.ptr(rect_output), n, k,
    rows, batch_size, out_dim);
}


void AggregateSpec::backward_task(const Task *task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  const AggregateSpecMeta* m = *((AggregateSpecMeta**)task->local_args);
  int n = ((AggregateSpec*)task->args)->n;
  float lambda_bal = ((AggregateSpec*)task->args)->lambda_bal;

  assert((int)regions.size() == n+5);
  assert((int)task->regions.size() == n+5);

  // get gate_pred, gate_assin, full_gate_grad, output_grad
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorRO<int, 2> acc_true_gate_assign(regions[2], FID_DATA);
  const AccessorWO<float, 2> acc_full_gate_grad(regions[3], FID_DATA);
  const AccessorRO<float, 2> acc_output_grad(regions[n+4], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_true_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_full_gate_grad = runtime->get_index_space_domain(
          ctx, task->regions[3].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[n+4].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_assign == rect_true_gate_assign);
  assert(batch_size == rect_full_gate_grad.hi[1] - rect_full_gate_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(k*batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;
  assert(n == rect_full_gate_grad.hi[0] - rect_full_gate_grad.lo[0] + 1);

  // get exp_preds
  float* exp_grads[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[4].region.get_index_space());
  exp_grads[0] = helperGetTensorPointerRW<float>(
    regions[4], task->regions[4], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+4].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
      regions[i+4], task->regions[i+4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  cudaStream_t stream;
  checkCUDA(create_stream(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  // call backward kernel
  cudaMemcpy(m->dev_region_ptrs, exp_grads, n*sizeof(float*), cudaMemcpyHostToDevice);

  aggspec_backward_kernel<<<GET_BLOCKS(batch_size*k*out_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*out_dim)), 0, stream>>>(
    m->dev_region_ptrs, acc_gate_assign.ptr(rect_gate_assign),
    acc_true_gate_assign.ptr(rect_true_gate_assign), acc_gate_pred.ptr(rect_gate_pred),
    acc_full_gate_grad.ptr(rect_full_gate_grad), acc_output_grad.ptr(rect_out_grad),
    n, k, rows, lambda_bal, batch_size, out_dim);
}


void AggregateSpec::forward(const FFModel& ff)
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
  IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+4], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+4].region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(n+2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


void AggregateSpec::backward(const FFModel& ff)
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

  IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);

  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // true gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[2], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[2].region));
  launcher.add_field(2, FID_DATA);

  // gate gradients full
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[3], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[3].region_grad));
  launcher.add_field(3, FID_DATA);

  // exp gradients
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i+4], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+4].region_grad));
    launcher.add_field(i+4, FID_DATA);
  }

  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(n+4, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


AggregateSpecMeta::AggregateSpecMeta(FFHandler handler, int n)
: OpMeta(handler)
{
  checkCUDA(cudaMalloc(&dev_region_ptrs, n*sizeof(float*)));
}
AggregateSpecMeta::~AggregateSpecMeta(void)
{
  checkCUDA(cudaFree(&dev_region_ptrs));
}


bool AggregateSpec::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}
