/* Copyright 2021 Facebook
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
#include <queue>
// #define MOE_SPEC_SCORE


// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]
void FFModel::top_k(const Tensor& input,
                    Tensor* outputs,
                    int k,
                    bool sorted,
                    const char *name)
{
  TopK* topk = new TopK(*this, input, k, sorted, name);
  layers.push_back(topk);
  assert(topk->numOutputs == 2);
  outputs[0] = topk->outputs[0];
  outputs[1] = topk->outputs[1];
}

TopK::TopK(FFModel& model,
           const Tensor& _input,
           int _k, bool _sorted,
           const char* name)
: Op(model, OP_TOPK, name, _input),
  k(_k), sorted(_sorted), profiling(model.config.profiling)
{
  numOutputs = 2; 
  outputs[0].numDim = inputs[0].numDim;
  outputs[1].numDim = inputs[0].numDim;
  outputs[0].adim[0] = k;
  outputs[1].adim[0] = k;
  for (int i = 1; i < inputs[0].numDim; i++) {
    outputs[0].adim[i] = outputs[1].adim[i] = inputs[0].adim[i];
  }
  numWeights = 0;
}

void TopK::create_weights(FFModel& model)
{
  // Do nothing
}

void TopK::create_output_and_partition(FFModel& model)
{
  int dim = inputs[0].numDim;
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
void TopK::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  dims[NDIM-1] = k;
  for (int i = 0; i < NDIM-1; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  outputs[1] = model.create_tensor<NDIM>(dims, DT_INT32, this);
  outputs[1].owner_op = this;
  outputs[1].owner_idx = 1;
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

OpMeta* TopK::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  TopK* topk = (TopK*) task->args;
  FFHandler handle = *((FFHandler*)task->local_args);
  TopKMeta* m = new TopKMeta(handle);
  m->profiling = topk->profiling;
  m->sorted = topk->sorted;
  return m;
}

void TopK::init(const FFModel& ff)
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
  IndexLauncher launcher(TOPK_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(TopK)), argmap,
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
  launcher.add_region_requirement(
    RegionRequirement(outputs[1].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[1].region));
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

void TopK::forward_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const TopK* topk = (const TopK*) task->args;
  const TopKMeta* m = *((TopKMeta**)task->local_args);
  Domain in1_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out1_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain out2_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());

  int in_cols = in1_domain.hi()[0] - in1_domain.lo()[0] + 1;
  int out1_cols = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  int out2_cols = out2_domain.hi()[0] - out2_domain.lo()[0] + 1;

  assert(out1_domain == out2_domain);
  for (int i = 1; i < in1_domain.get_dim(); i++) {
    assert(in1_domain.lo()[i] == out1_domain.lo()[i]);
    assert(in1_domain.hi()[i] == out1_domain.hi()[i]);
  }
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* value_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int* index_ptr = helperGetTensorPointerWO<int>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int length = in1_domain.hi()[0] - in1_domain.lo()[0] + 1; // n
  int k = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  size_t batch_size = in1_domain.get_volume() / length;

  std::priority_queue<std::pair<float, int>> q;
  for(int i = 0; i < batch_size; i++) {
    q = std::priority_queue<std::pair<float, int>>();
    for (int j = 0; j < length; j++) {
      q.push(std::pair<float, int>(in_ptr[i*length+j], j));
    }

    for(int j = 0; j < k; j++) {
      value_ptr[i*k+j] = q.top().first;
      index_ptr[i*k+j] = q.top().second;
      q.pop();
    }
  }

  // for(int i = 0; i < batch_size; i++) {
  //   printf("input ");
  //   for(int j = 0; j < length; j++) {
  //     printf("%.2f ", in_ptr[i*length+j]);
  //   }
  //   printf("------ ");
  //   for(int j = 0; j < k; j++) {
  //     printf("%d (%.2f) ", index_ptr[i*k+j], value_ptr[i*k+j]);
  //   }
  //   printf("\n");
  // }

}

void TopK::forward(const FFModel& ff)
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
  IndexLauncher launcher(TOPK_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
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
  launcher.add_region_requirement(
    RegionRequirement(outputs[1].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[1].region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): out1_grad
  regions[1](I): out2
  regions[2](I/0): in_grad
*/
void TopK::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime* runtime)
{
  return;
  // //const TopK* topk = (const TopK*) task->args;
  // const TopKMeta* m = *((TopKMeta**) task->local_args);
  // assert(regions.size() == 3);
  // Domain out1_domain = runtime->get_index_space_domain(
  //   ctx, task->regions[0].region.get_index_space());
  // Domain out2_domain = runtime->get_index_space_domain(
  //   ctx, task->regions[1].region.get_index_space());
  // Domain in_domain = runtime->get_index_space_domain(
  //   ctx, task->regions[2].region.get_index_space());
  // assert(out1_domain == out2_domain);
  // for (int i = 1; i < in_domain.get_dim(); i++) {
  //   assert(in_domain.lo()[i] == out1_domain.lo()[i]);
  //   assert(in_domain.hi()[i] == out1_domain.hi()[i]);
  // }
  // const float* value_grad_ptr = helperGetTensorPointerRO<float>(
  //   regions[0], task->regions[0], FID_DATA, ctx, runtime);
  // const int* indices_ptr = helperGetTensorPointerRO<int>(
  //   regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // float* in_grad_ptr = helperGetTensorPointerRW<float>(
  //   regions[2], task->regions[2], FID_DATA, ctx, runtime);
  //
  // cudaStream_t stream;
  // checkCUDA(get_legion_stream(&stream));
  //
  // cudaEvent_t t_start, t_end;
  // if (m->profiling) {
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);
  // }
  // int length = in_domain.hi()[0] - in_domain.lo()[0] + 1;
  // int k = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  // size_t batch_size = in_domain.get_volume() / length;
  // backward_kernel(m, value_grad_ptr, indices_ptr, in_grad_ptr,
  //     batch_size, length, k, stream);

  // TODO: missing profiling here
}

void TopK::backward(const FFModel& ff)
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

  IndexLauncher launcher(TOPK_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): value_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): indices
  launcher.add_region_requirement(
    RegionRequirement(outputs[1].part, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[1].region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

TopKMeta::TopKMeta(FFHandler handler)
: OpMeta(handler)
{
}

bool TopK::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  // To be implemented
  // assert(false);
  return true;
}
