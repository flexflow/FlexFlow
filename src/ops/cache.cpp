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
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

template <typename T>
void Cache::cache_forward(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Cache *c = ((Arg *)(task->args))->cache;
  CacheMeta const *m = *((CacheMeta **)task->local_args);
  int batch_ctr = ((Arg *)(task->args))->batch_ctr;
  T **batch_ptrs = (T **)c->batch_ptrs;
  T *output_ptr = helperGetTensorPointerWO<T>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);

  // TODO: Check why cublas/cudnn stream is needed here
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  checkCUDA(hipMemcpy(output_ptr,
                      batch_ptrs[batch_ctr],
                      c->inputs[0]->get_volume() * sizeof(T),
                      hipMemcpyHostToDevice));
}

template <typename T>
float Cache::cache_update(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Cache *c = ((Arg *)(task->args))->cache;
  int batch_ctr = ((Arg *)(task->args))->batch_ctr;
  CacheMeta *m = *((CacheMeta **)task->local_args);

  T const *input_ptr = helperGetTensorPointerRW<T>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  T *host_input = (T *)c->batch_cmp;
  checkCUDA(hipMemcpy(host_input,
                      input_ptr,
                      c->inputs[0]->get_volume() * sizeof(T),
                      hipMemcpyDeviceToHost));
  float cache_score = c->score_f(&m->cache_score,
                                 host_input,
                                 c->batch_ptrs[batch_ctr],
                                 c->inputs[0]->get_volume());
  memcpy(c->batch_ptrs[batch_ctr],
         host_input,
         c->inputs[0]->get_volume() * sizeof(T));
  return cache_score;
}

CacheMeta::CacheMeta(FFHandler handler) : OpMeta(handler) {}

template void
    Cache::cache_forward<float>(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime);
template void
    Cache::cache_forward<int32_t>(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);

template float
    Cache::cache_update<float>(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime);
template float
    Cache::cache_update<int32_t>(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime);

}; // namespace FlexFlow
