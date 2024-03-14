/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/ops/cache.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::PointInRectIterator;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

// Moving average over batches: 1 if batch is perfectly cached, 0 else.
template <typename T>
float default_score(float *cached_score,
                    void const *input,
                    void const *cached,
                    int vol) {
  float gamma = 0.99f;
  *cached_score *= gamma;
  float w = 1.0f - gamma;
  T *cast_input = (T *)input;
  T *cast_cached = (T *)cached;
  for (int i = 0; i < vol; i++) {
    if (cast_input[i] != cast_cached[i]) {
      return *cached_score;
    }
  }
  *cached_score += w;
  return *cached_score;
}

Tensor FFModel::cache(
    Tensor const &input,
    int num_batches,
    std::function<float(float *, void const *, void const *, int)> score_f,
    char const *name) {
  assert(false);
#ifdef DEADCODE
  if (!score_f) {
    switch (input->data_type) {
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
  Cache *cache = new Cache(*this, input, num_batches, score_f, name);
  layers.push_back(cache);
  return cache->outputs[0];
#endif
}

Cache::Cache(
    FFModel &model,
    ParallelTensor const &_input,
    int _num_batches,
    std::function<float(float *, void const *, void const *, int)> &_score_f,
    char const *name)
    : Op(model,
         OP_CACHE,
         DT_FLOAT,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outptus*/,
         _input),
      num_batches(_num_batches), score_f(_score_f) {
  load_cached = false;
  batch_ctr = 0;

  int num_dim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dim, dims, DT_FLOAT, this);

  numWeights = 0;
}

Cache::~Cache() {
  for (int i = 0; i < num_batches; i++) {
    free(batch_ptrs[i]);
  }
  free(batch_ptrs);
  free(batch_cmp);
}

template <typename T>
void cache_init(Cache *cache, size_t vol) {
  // init pointer array
  cache->batch_ptrs = (void **)malloc(cache->num_batches * sizeof(T *));
  for (int i = 0; i < cache->num_batches; i++) {
    cache->batch_ptrs[i] = malloc(vol * sizeof(T));
  }
  cache->batch_cmp = malloc(vol * sizeof(T));
}

void Cache::init(FFModel const &ff) {
  size_t vol = inputs[0]->get_volume();
  switch (inputs[0]->data_type) {
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
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(CACHE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Cache)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Cache::init_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  Cache *c = (Cache *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  CacheMeta *m = new CacheMeta(handle);
  m->cache_score = 0.0f;
  m->profiling = c->profiling;
  m->inference_debugging = c->inference_debugging;
  std::strcpy(m->op_name, c->name);
  m->layer_guid = c->layer_guid;
  return m;
}

void Cache::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  Arg arg = {this, batch_ctr};
  // Launch update task
  IndexLauncher launcher_update(CACHE_UPDATE_TASK_ID,
                                parallel_is,
                                TaskArgument(&arg, sizeof(Arg)),
                                argmap,
                                Predicate::TRUE_PRED,
                                false /*must*/,
                                0 /*mapper_id*/,
                                FFConfig::get_hash_id(std::string(name)));
  launcher_update.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                           0 /*projection id*/,
                                                           READ_WRITE,
                                                           EXCLUSIVE,
                                                           inputs[0]->region));
  launcher_update.add_field(0, FID_DATA);
  FutureMap score_fm = runtime->execute_index_space(ctx, launcher_update);
  // add score futures to Cache future vector attribute
  score_futures.clear();
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    for (PointInRectIterator<DIM> it(rect); it(); it++)                        \
      score_futures.push_back(score_fm[*it]);                                  \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  // Launch forward task
  if (load_cached) {
    IndexLauncher launcher_fwd(CACHE_FWD_TASK_ID,
                               parallel_is,
                               TaskArgument(&arg, sizeof(Arg)),
                               argmap,
                               Predicate::TRUE_PRED,
                               false /*must*/,
                               0 /*mapper_id*/,
                               FFConfig::get_hash_id(std::string(name)));
    launcher_fwd.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                          0 /*projection id*/,
                                                          WRITE_ONLY,
                                                          EXCLUSIVE,
                                                          outputs[0]->region));
    launcher_fwd.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher_fwd);
  }
  batch_ctr = (batch_ctr + 1) % num_batches;
}

void Cache::forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  Cache *c = ((Arg *)(task->args))->cache;
  assert((int)regions.size() == 1);
  assert((int)task->regions.size() == 1);

  switch (c->inputs[0]->data_type) {
    case DT_FLOAT:
      Cache::cache_forward<float>(task, regions, ctx, runtime);
      break;
    case DT_INT32:
      Cache::cache_forward<int32_t>(task, regions, ctx, runtime);
      break;
    default:
      assert(false && "unsupported data type");
      break;
  }
}

void Cache::backward(FFModel const &ff) {
  // Do nothing
}

void Cache::use_cached(bool c) {
  load_cached = c;
}

float Cache::update_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  Cache *c = ((Arg *)(task->args))->cache;
  switch (c->inputs[0]->data_type) {
    case DT_FLOAT:
      return Cache::cache_update<float>(task, regions, ctx, runtime);
    case DT_INT32:
      return Cache::cache_update<int32_t>(task, regions, ctx, runtime);
    default:
      assert(false && "unsupported data type");
      return -1.0f;
  }
}

bool Cache::measure_operator_cost(Simulator *sim,
                                  MachineView const &mv,
                                  CostMetrics &cost_metrics) const {
  // TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return false;
}

}; // namespace FlexFlow
