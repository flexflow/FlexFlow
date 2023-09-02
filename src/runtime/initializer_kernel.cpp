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

#include "flexflow/accessor.h"
#include "flexflow/initializer.h"
#include "flexflow/model.h"
#include "flexflow/utils/hip_helper.h"
#include <ctime>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <random>

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
void UniformInitializer::init_task(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {

  assert(regions.size() == task->regions.size());
  UniformInitializer *initializer = (UniformInitializer *)task->args;
  // Assume the data type is float
  assert(initializer->data_type == DT_FLOAT);
  hiprandGenerator_t gen;
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hiprandSetStream(gen, stream);
  // fprintf(stderr, "seed = %d\n", initializer->seed);

  for (size_t i = 0; i < regions.size(); i++) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    float *w;
    switch (domain.get_dim()) {
      case 0: {
        // Do not support 0-dim parameters
        assert(false);
        break;
      }
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<float, DIM> accW(regions[i],                               \
                                     task->regions[i],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     false /*readOutput*/);                    \
    w = accW.ptr;                                                              \
    break;                                                                     \
  }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default: {
        assert(false);
        break;
      }
    }
    hiprandSetPseudoRandomGeneratorSeed(gen, initializer->seed);
    checkCUDA(hiprandGenerateUniform(gen, w, domain.get_volume()));
    hipLaunchKernelGGL(scale_kernel,
                       GET_BLOCKS(domain.get_volume()),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       w,
                       domain.get_volume(),
                       initializer->min_val,
                       initializer->max_val);
  }
  checkCUDA(hipDeviceSynchronize());
  hiprandDestroyGenerator(gen);
}

template <int NDIM>
void init_task_inner(Task const *task,
                     std::vector<PhysicalRegion> const &regions,
                     Context ctx,
                     Runtime *runtime,
                     Domain const &domain,
                     float *&w,
                     float &scale) {
  TensorAccessorW<float, NDIM> accW(regions[0],
                                    task->regions[0],
                                    FID_DATA,
                                    ctx,
                                    runtime,
                                    false /*readOutput*/);
  w = accW.ptr;
  // reference: tensorflow code for computing fan_in/fan_out
  // https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/init_ops.py#L1415-L1439
  int num_dim = domain.get_dim();
  coord_t receptive_field_size = 1;
  for (int i = 2; i < num_dim; i++) {
    receptive_field_size *= (accW.rect.hi[i] - accW.rect.lo[i] + 1);
  }
  coord_t c_in = accW.rect.hi[1] - accW.rect.lo[1] + 1;
  coord_t c_out = accW.rect.hi[0] - accW.rect.lo[0] + 1;
  coord_t fan_in = c_in * receptive_field_size;
  coord_t fan_out = c_out * receptive_field_size;
  scale = sqrt(6.0 / (fan_in + fan_out));
}

void GlorotUniform::init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  GlorotUniform const *gu = (GlorotUniform const *)task->args;
  assert(gu->data_type == DT_FLOAT);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  float *w = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  hiprandGenerator_t gen;
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCURAND(hiprandSetStream(gen, stream));

  GlorotUniform *initializer = (GlorotUniform *)task->args;
  hiprandSetPseudoRandomGeneratorSeed(gen, initializer->seed);
  fprintf(stderr, "seed = %d scale = %.4lf\n", initializer->seed, gu->scale);
  checkCUDA(hiprandGenerateUniform(gen, w, domain.get_volume()));
  hipLaunchKernelGGL(scale_kernel,
                     GET_BLOCKS(domain.get_volume()),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     w,
                     domain.get_volume(),
                     -gu->scale,
                     gu->scale);
  checkCUDA(hipDeviceSynchronize());
  hiprandDestroyGenerator(gen);
}

void NormInitializer::init_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  float *w;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<float, DIM> accW(regions[0],                               \
                                     task->regions[0],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     false /*readOutput*/);                    \
    w = accW.ptr;                                                              \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  hiprandGenerator_t gen;
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCURAND(hiprandSetStream(gen, stream));

  NormInitializer *initializer = (NormInitializer *)task->args;
  // fprintf(stderr, "seed = %d\n", initializer->seed);
  hiprandSetPseudoRandomGeneratorSeed(gen, initializer->seed);
  // fprintf(stderr, "domain.volume() = %zu mean(%.4lf) var(%.4lf)\n",
  //     domain.get_volume(), initializer->mean, initializer->stddev);
  //  FIXME: it seems hiprand has an internal bug with volume < 4
  //  double check this later
  if (domain.get_volume() < 4) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(initializer->mean,
                                                 initializer->stddev);
    float *w_dram = (float *)malloc(domain.get_volume() * sizeof(float));
    for (size_t i = 0; i < domain.get_volume(); i++) {
      w_dram[i] = distribution(generator);
    }
    checkCUDA(hipMemcpy(
        w, w_dram, sizeof(float) * domain.get_volume(), hipMemcpyHostToDevice));
    checkCUDA(hipDeviceSynchronize());
    free(w_dram);
  } else {
    checkCURAND(hiprandGenerateNormal(
        gen, w, domain.get_volume(), initializer->mean, initializer->stddev));
    checkCUDA(hipDeviceSynchronize());
  }
  hiprandDestroyGenerator(gen);
}

void ZeroInitializer::init_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  ZeroInitMeta *meta = (ZeroInitMeta *)task->args;
  assert(meta->num_regions == regions.size());
  assert(regions.size() == task->regions.size());
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  for (size_t i = 0; i < regions.size(); i++) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    if (meta->data_types[i] == DT_HALF) {
      half *w = helperGetTensorPointerWO<half>(
          regions[i], task->regions[i], FID_DATA, ctx, runtime);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(assign_kernel<half>),
                         GET_BLOCKS(domain.get_volume()),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         w,
                         domain.get_volume(),
                         0.0f);
    } else if (meta->data_types[i] == DT_FLOAT) {
      float *w = helperGetTensorPointerWO<float>(
          regions[i], task->regions[i], FID_DATA, ctx, runtime);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(assign_kernel<float>),
                         GET_BLOCKS(domain.get_volume()),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         w,
                         domain.get_volume(),
                         0.0f);
    } else if (meta->data_types[i] == DT_INT32) {
      int32_t *w = helperGetTensorPointerWO<int32_t>(
          regions[i], task->regions[i], FID_DATA, ctx, runtime);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(assign_kernel<int32_t>),
                         GET_BLOCKS(domain.get_volume()),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         w,
                         domain.get_volume(),
                         0);
    } else if (meta->data_types[i] == DT_INT64) {
      int64_t *w = helperGetTensorPointerWO<int64_t>(
          regions[i], task->regions[i], FID_DATA, ctx, runtime);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(assign_kernel<int64_t>),
                         GET_BLOCKS(domain.get_volume()),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         w,
                         domain.get_volume(),
                         0);
    } else {
      assert(false && "Unsupported data type in Zero Initializer");
    }
  }
  checkCUDA(hipDeviceSynchronize());
}

void ConstantInitializer::init_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  ConstantInitializer *initializer = (ConstantInitializer *)task->args;
  assert(regions.size() == task->regions.size());
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  for (size_t i = 0; i < regions.size(); i++) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    switch (initializer->data_type) {
      case DT_FLOAT: {
        float *w = helperGetTensorPointerWO<float>(
            regions[i], task->regions[i], FID_DATA, ctx, runtime);
        hipLaunchKernelGGL(assign_kernel,
                           GET_BLOCKS(domain.get_volume()),
                           CUDA_NUM_THREADS,
                           0,
                           stream,
                           w,
                           domain.get_volume(),
                           initializer->float_value);
        break;
      }
      case DT_INT64: {
        int64_t *w = helperGetTensorPointerWO<int64_t>(
            regions[i], task->regions[i], FID_DATA, ctx, runtime);
        hipLaunchKernelGGL(assign_kernel,
                           GET_BLOCKS(domain.get_volume()),
                           CUDA_NUM_THREADS,
                           0,
                           stream,
                           w,
                           domain.get_volume(),
                           initializer->int64_value);
        break;
      }
      case DT_INT32: {
        int *w = helperGetTensorPointerWO<int>(
            regions[i], task->regions[i], FID_DATA, ctx, runtime);
        hipLaunchKernelGGL(assign_kernel,
                           GET_BLOCKS(domain.get_volume()),
                           CUDA_NUM_THREADS,
                           0,
                           stream,
                           w,
                           domain.get_volume(),
                           initializer->int32_value);
        break;
      }
      default: {
        assert(false && "Unsupported Initialzier Type");
      }
    }
  }
  checkCUDA(hipDeviceSynchronize());
}

}; // namespace FlexFlow
