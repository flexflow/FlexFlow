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


Tensor FFModel::cache(const Tensor& input, int num_batches, const char* name)
{
  Cache* cache = new Cache(*this, input, num_batches, name);
  layers.push_back(cache);
  return cache->outputs[0];
}


Cache::Cache(FFModel& model,
            const Tensor& _input,
            int _num_batches,
            const char* name)
: Op(model, OP_CACHE, name, _input),
  num_batches(_num_batches),
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
  return NULL;
}


template <typename T>
void cache_init(Cache* cache, size_t vol) {
  // init pointer array
  cache->batch_ptrs = (void**)malloc(cache->num_batches*sizeof(T*));
  for(int i = 0; i < cache->num_batches; i++)
    cache->batch_ptrs[i] = malloc(vol*sizeof(T));
}


void Cache::use_cached(bool cached) {
  load_cached = cached;
}


void Cache::init(const FFModel& ff)
{
  size_t vol = inputs[0].get_volume();
  switch(inputs[0].data_type)
  {
    case DT_FLOAT:
      cache_init<float>(this, vol);
      break;
    // case DT_DOUBLE:
    //   cache_forward<double>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    case DT_INT32:
      cache_init<int32_t>(this, vol);
      break;
    // case DT_INT64:
    //   cache_forward<int64_t>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    // case DT_BOOLEAN:
    //   cache_forward<bool>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    default:
      assert(false && "unsupported data type");
      break;
  }
}


template <typename T>
void cache_forward(const FFModel* ff, int num_batches, int batch, bool load_cached,
  void** &batch_ptrs, std::vector<int>& dims, Tensor& input, Tensor& output)
{
  T** cast_batch_ptrs = (T**)batch_ptrs;
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif

  // load or store
  if(load_cached) {
    output.set_tensor<T>(ff, dims, cast_batch_ptrs[batch], comm_type);
  }
  else {
    input.get_tensor<T>(ff, cast_batch_ptrs[batch], comm_type);
  }
}


void Cache::forward(const FFModel& ff)
{
  int num_dim = inputs[0].numDim;
  std::vector<int> dims(num_dim);
  for(int i = 0; i < num_dim; i++) dims[i] = inputs[0].adim[num_dim-1-i];

  switch(inputs[0].data_type)
  {
    case DT_FLOAT:
      cache_forward<float>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
        dims, inputs[0], outputs[0]);
      break;
    // case DT_DOUBLE:
    //   cache_forward<double>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    case DT_INT32:
      cache_forward<int32_t>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
        dims, inputs[0], outputs[0]);
      break;
    // case DT_INT64:
    //   cache_forward<int64_t>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    // case DT_BOOLEAN:
    //   cache_forward<bool>(&ff, num_batches, batch_ctr, load_cached, batch_ptrs,
    //     dims, inputs[0], outputs[0]);
    //   break;
    default:
      assert(false && "unsupported data type");
      break;
  }

  batch_ctr = (batch_ctr+1)%num_batches;
}


void Cache::backward(const FFModel& ff)
{
  // Do nothing
}


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
