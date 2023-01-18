/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/initializer.h"
#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

Initializer::Initializer(void) {}

Initializer::~Initializer(void) {}

GlorotUniform::GlorotUniform(int _seed) : Initializer(), seed(_seed) {}

GlorotUniform::~GlorotUniform(void) {}

void GlorotUniform::init(FFModel const *ff, const ParallelTensor p) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  int dim = p->num_dims - 1;
  while (p->dims[dim].is_replica_dim) {
    dim--;
  }
  assert(dim >= 1);
  // reference: tensorflow code for computing fan_in/fan_out
  // https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/init_ops.py#L1415-L1439
  coord_t c_out = p->dims[dim--].size;
  coord_t c_in = p->dims[dim--].size;
  coord_t receptive_field_size = 1;
  while (dim >= 0) {
    receptive_field_size *= p->dims[dim--].size;
  }
  coord_t fan_in = c_in * receptive_field_size;
  coord_t fan_out = c_out * receptive_field_size;
  scale = sqrt(6.0f / (fan_in + fan_out));
  this->data_type = p->data_type;
  if (p->sync_type == ParameterSyncType::PS) {
    assert(p->num_dims >= 2);
    TaskLauncher launcher(GLOROT_INIT_TASK_ID,
                          TaskArgument(this, sizeof(GlorotUniform)));
    // regions[0]: p->region
    launcher.add_region_requirement(
        RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // assert(p->owner_op != NULL);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    assert(p->num_dims >= 2);
    ArgumentMap argmap;
    IndexLauncher launcher(GLOROT_INIT_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(GlorotUniform)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false,
                           0,
                           p->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(false);
  }
}

ZeroInitializer::ZeroInitializer(void) : Initializer() {}

ZeroInitializer::~ZeroInitializer(void) {}

void ZeroInitializer::init(FFModel const *ff, const ParallelTensor p) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  if (p->sync_type == ParameterSyncType::PS) {
    ZeroInitMeta meta;
    meta.num_regions = 1;
    meta.data_types[0] = p->data_type;
    TaskLauncher launcher(ZERO_INIT_TASK_ID,
                          TaskArgument(&meta, sizeof(ZeroInitMeta)));
    // regions[0]: p->region
    launcher.add_region_requirement(
        RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // assert(p->owner_op != NULL);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    ZeroInitMeta meta;
    meta.num_regions = 1;
    meta.data_types[0] = p->data_type;
    IndexLauncher launcher(ZERO_INIT_TASK_ID,
                           p->parallel_is,
                           TaskArgument(&meta, sizeof(ZeroInitMeta)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false,
                           0,
                           p->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(false);
  }
}

void ZeroInitializer::init_task_cpu(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(regions.size() == task->regions.size());
  for (size_t i = 0; i < regions.size(); i++) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    float *w;
    switch (domain.get_dim()) {
      case 0: {
        // Do not support 0-dim parameters for now
        assert(false);
        break;
      }
      case 1: {
        AccessorWO<float, 1> const accW(regions[i], FID_DATA);
        Rect<1> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      case 2: {
        AccessorWO<float, 2> const accW(regions[i], FID_DATA);
        Rect<2> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      case 3: {
        AccessorWO<float, 3> const accW(regions[i], FID_DATA);
        Rect<3> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
    for (size_t i = 0; i < domain.get_volume(); i++) {
      w[i] = 0.0f;
    }
  }
}

UniformInitializer::UniformInitializer(int _seed, float _min, float _max)
    : Initializer(), seed(_seed), min_val(_min), max_val(_max) {}

UniformInitializer::~UniformInitializer(void) {}

void UniformInitializer::init(FFModel const *ff, const ParallelTensor p) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  this->data_type = p->data_type;
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(UNIFORM_INIT_TASK_ID,
                          TaskArgument(this, sizeof(UniformInitializer)));
    // regions[0]: p->region
    launcher.add_region_requirement(
        RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // assert(p->owner_op != NULL);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    IndexLauncher launcher(UNIFORM_INIT_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(UniformInitializer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false,
                           0,
                           p->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(false);
  }
}

NormInitializer::NormInitializer(int _seed, float _mean, float _stddev)
    : seed(_seed), mean(_mean), stddev(_stddev) {}

NormInitializer::~NormInitializer(void) {}

void NormInitializer::init(FFModel const *ff, const ParallelTensor p) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  this->data_type = p->data_type;
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(NORMAL_INIT_TASK_ID,
                          TaskArgument(this, sizeof(NormInitializer)));
    // regions[0]: p->region
    launcher.add_region_requirement(
        RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // assert(p->owner_op != NULL);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    IndexLauncher launcher(NORMAL_INIT_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(NormInitializer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false,
                           0,
                           p->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(false);
  }
}

// ConstantInitializer
ConstantInitializer::ConstantInitializer(float _value)
    : Initializer(), data_type(DT_FLOAT), float_value(_value) {}

ConstantInitializer::ConstantInitializer(int64_t _value)
    : Initializer(), data_type(DT_INT64), int64_value(_value) {}

ConstantInitializer::ConstantInitializer(int _value)
    : Initializer(), data_type(DT_INT32), int32_value(_value) {}

ConstantInitializer::~ConstantInitializer(void) {}

void ConstantInitializer::init(FFModel const *ff, const ParallelTensor p) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  assert(p->data_type == this->data_type);
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(CONSTANT_INIT_TASK_ID,
                          TaskArgument(this, sizeof(ConstantInitializer)));
    // regions[0]: p->region
    launcher.add_region_requirement(
        RegionRequirement(p->region, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // assert(p->owner_op != NULL);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    IndexLauncher launcher(CONSTANT_INIT_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(ConstantInitializer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false,
                           0,
                           p->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, WRITE_ONLY, EXCLUSIVE, p->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(false);
  }
}

void ConstantInitializer::init_task_cpu(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  ConstantInitializer *initializer = (ConstantInitializer *)task->args;
  assert(regions.size() == task->regions.size());
  for (size_t i = 0; i < regions.size(); i++) {
    // Currently only support Float
    assert(initializer->data_type == DT_FLOAT);
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[i].region.get_index_space());
    float *w;
    switch (domain.get_dim()) {
      case 0: {
        // Do not support 0-dim parameters for now
        assert(false);
        break;
      }
      case 1: {
        AccessorWO<float, 1> const accW(regions[i], FID_DATA);
        Rect<1> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      case 2: {
        AccessorWO<float, 2> const accW(regions[i], FID_DATA);
        Rect<2> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      case 3: {
        AccessorWO<float, 3> const accW(regions[i], FID_DATA);
        Rect<3> rect = runtime->get_index_space_domain(
            ctx, task->regions[i].region.get_index_space());
        assert(accW.accessor.is_dense_arbitrary(rect));
        w = accW.ptr(rect);
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
    for (size_t i = 0; i < domain.get_volume(); i++) {
      w[i] = initializer->float_value;
    }
  }
}

}; // namespace FlexFlow
