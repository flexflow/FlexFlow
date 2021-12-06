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

#include <hip/hip_runtime.h>
#include "flexflow/ops/cache.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;
using Legion::Memory;

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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  hipMemcpy(output_ptr, batch_ptrs[batch_ctr], c->inputs[0]->get_volume()*sizeof(T), hipMemcpyHostToDevice);
}


void Cache::forward_task(const Task *task,
                        const std::vector<PhysicalRegion>& regions,
                        Context ctx, Runtime* runtime)
{
  Cache* c = ((Arg*)(task->args))->cache;
  assert((int)regions.size() == 1);
  assert((int)task->regions.size() == 1);

  switch(c->inputs[0]->data_type)
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
  hipMemcpy(host_input, input_ptr, c->inputs[0]->get_volume()*sizeof(T), hipMemcpyDeviceToHost);
  float cache_score = c->score_f(&m->cache_score, host_input,
    c->batch_ptrs[batch_ctr], c->inputs[0]->get_volume());
  memcpy(c->batch_ptrs[batch_ctr], host_input, c->inputs[0]->get_volume()*sizeof(T));
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
  switch(c->inputs[0]->data_type)
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

CacheMeta::CacheMeta(FFHandler handler)
: OpMeta(handler)
{}

bool Cache::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics) const
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}

}; // namespace FlexFlow
