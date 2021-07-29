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

// Moving average over batches: 1 if batch is perfectly cached, 0 else.
template <typename T>
float default_score(float* cached_score,
                    const void* input,
                    const void* cached,
                    int vol) {
  float gamma = 0.99f;
  *cached_score *= gamma;
  float w = 1.0f-gamma;
  T* cast_input = (T*)input;
  T* cast_cached = (T*)cached;
  for(int i = 0; i < vol; i++) {
    if(cast_input[i] != cast_cached[i]) {
      return *cached_score;
    }
  }
  *cached_score += w;
  return *cached_score;
}


struct Arg {
  Cache* cache;
  int batch_ctr;
};


Tensor FFModel::cache(const Tensor& input, int num_batches,
  std::function<float(float*,const void*,const void*,int)> score_f, const char* name)
{
  if(!score_f) {
    switch(input.data_type) {
      case DT_FLOAT:
        score_f = default_score<float>;
        break;
      case DT_INT32:
        score_f = default_score<int32_t>;
        break;
      default:
        assert(false && "unsupported data type");
        break;
    }
  }
  Cache* cache = new Cache(*this, input, num_batches, score_f, name);
  layers.push_back(cache);
  return cache->outputs[0];
}


Cache::Cache(FFModel& model,
            const Tensor& _input,
            int _num_batches,
            std::function<float(float*,const void*,const void*,int)> &_score_f,
            const char* name)
: Op(model, OP_CACHE, name, _input),
  num_batches(_num_batches),
  score_f(_score_f),
  profiling(model.config.profiling)
{
  load_cached = false;
  batch_ctr = 0;

  int num_dim = inputs[0].numDim;
  outputs[0].numDim = num_dim;
  for(int i = 0; i < num_dim; i++)
    outputs[0].adim[i] = inputs[0].adim[i];

  numWeights = 0;
}


Cache::~Cache() {
  for(int i = 0; i < num_batches; i++) free(batch_ptrs[i]);
  free(batch_ptrs);
  free(batch_cmp);
}


void Cache::create_weights(FFModel& model)
{
  // Do nothing
}


void Cache::create_output_and_partition(FFModel& model)
{
  // Retrieve the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);

  // create output
  int num_dim = inputs[0].numDim;
  assert(num_dim == domain.get_dim());
  int dims[num_dim];
  for (int i = 0; i < num_dim; i++)
    dims[i] = inputs[0].adim[num_dim-1-i];

  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> part_rect = domain; \
      outputs[0] = model.create_tensor<DIM>(dims, inputs[0].data_type, this); \
      outputs[0].owner_op = this; \
      outputs[0].owner_idx = 0; \
      Rect<DIM> input_rect = runtime->get_index_partition_color_space( \
          ctx, inputs[0].part.get_index_partition()); \
      if (input_rect == part_rect) { \
        input_lps[0] = inputs[0].part; \
        input_grad_lps[0] = inputs[0].part_grad; \
      } else { \
        model.create_disjoint_partition<DIM>(inputs[0], \
            IndexSpaceT<DIM>(task_is), input_lps[0], input_grad_lps[0]); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      fprintf(stderr, "Unsupported dimension number");
      assert(false);
    }
  }
}


OpMeta* Cache::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  Cache* c = (Cache*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  CacheMeta* m = new CacheMeta(handle);
  m->cache_score = 0.0f;
  m->profiling = c->profiling;
  return m;
}


template <typename T>
void cache_init(Cache* cache, size_t vol) {
  // init pointer array
  cache->batch_ptrs = (void**)malloc(cache->num_batches*sizeof(T*));
  for(int i = 0; i < cache->num_batches; i++)
    cache->batch_ptrs[i] = malloc(vol*sizeof(T));
  cache->batch_cmp = malloc(vol*sizeof(T));
}


void Cache::init(const FFModel& ff)
{
  size_t vol = inputs[0].get_volume();
  switch(inputs[0].data_type)
  {
    case DT_FLOAT:
      cache_init<float>(this, vol);
      break;
    case DT_INT32:
      cache_init<int32_t>(this, vol);
      break;
    default:
      assert(false && "unsupported data type");
      break;
  }
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
  IndexLauncher launcher(CACHE_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(Cache)), argmap,
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


template <typename T>
void cache_forward(const Task *task,
                  const std::vector<PhysicalRegion>& regions,
                  Context ctx, Runtime* runtime)
{
  Cache* c = ((Arg*)(task->args))->cache;
  const CacheMeta* m = *((CacheMeta**)task->local_args);
  int batch_ctr = ((Arg*)(task->args))->batch_ctr;
  T** batch_ptrs = (T**)c->batch_ptrs;
  T* output_ptr = helperGetTensorPointerWO<T>(regions[0], task->regions[0],
    FID_DATA, ctx, runtime);

  // TODO: Check why cublas/cudnn stream is needed here
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  cudaMemcpy(output_ptr, batch_ptrs[batch_ctr], c->inputs[0].get_volume()*sizeof(T), cudaMemcpyHostToDevice);
}


void Cache::forward_task(const Task *task,
                        const std::vector<PhysicalRegion>& regions,
                        Context ctx, Runtime* runtime)
{
  Cache* c = ((Arg*)(task->args))->cache;
  assert((int)regions.size() == 1);
  assert((int)task->regions.size() == 1);

  switch(c->inputs[0].data_type)
  {
    case DT_FLOAT:
      cache_forward<float>(task, regions, ctx, runtime);
      break;
    case DT_INT32:
      cache_forward<int32_t>(task, regions, ctx, runtime);
      break;
    default:
      assert(false && "unsupported data type");
      break;
  }
}


template <typename T>
float cache_update(const Task *task,
                  const std::vector<PhysicalRegion>& regions,
                  Context ctx, Runtime* runtime)
{
  Cache* c = ((Arg*)(task->args))->cache;
  int batch_ctr = ((Arg*)(task->args))->batch_ctr;
  CacheMeta* m = *((CacheMeta**)task->local_args);

  const T* input_ptr = helperGetTensorPointerRW<T>(regions[0], task->regions[0],
    FID_DATA, ctx, runtime);
  T* host_input = (T*) c->batch_cmp;
  cudaMemcpy(host_input, input_ptr, c->inputs[0].get_volume()*sizeof(T), cudaMemcpyDeviceToHost);
  float cache_score = c->score_f(&m->cache_score, host_input,
    c->batch_ptrs[batch_ctr], c->inputs[0].get_volume());
  memcpy(c->batch_ptrs[batch_ctr], host_input, c->inputs[0].get_volume()*sizeof(T));
  return cache_score;
}


void Cache::use_cached(bool c) {
  load_cached = c;
}


float Cache::update_task(const Task *task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime)
{
  Cache* c = ((Arg*)(task->args))->cache;
  switch(c->inputs[0].data_type)
  {
    case DT_FLOAT:
      return cache_update<float>(task, regions, ctx, runtime);
    case DT_INT32:
      return cache_update<int32_t>(task, regions, ctx, runtime);
    default:
      assert(false && "unsupported data type");
      return -1.0f;
  }
}


void Cache::forward(const FFModel& ff)
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

  Arg arg = {this, batch_ctr};
  // Launch update task
  IndexLauncher launcher_update(CACHE_UPDATE_TASK_ID, task_is,
                         TaskArgument(&arg, sizeof(Arg)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher_update.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher_update.add_field(0, FID_DATA);
  FutureMap score_fm = runtime->execute_index_space(ctx, launcher_update);
  // add score futures to Cache future vector attribute
  score_futures.clear();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) \
        score_futures.push_back(score_fm[*it]); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  // Launch forward task
  if(load_cached) {
    IndexLauncher launcher_fwd(CACHE_FWD_TASK_ID, task_is,
                           TaskArgument(&arg, sizeof(Arg)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(name)));
    launcher_fwd.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
    launcher_fwd.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher_fwd);
  }
  batch_ctr = (batch_ctr+1)%num_batches;
}


void Cache::backward(const FFModel& ff)
{
  // Do nothing
}


CacheMeta::CacheMeta(FFHandler handler)
: OpMeta(handler)
{}

bool Cache::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}
