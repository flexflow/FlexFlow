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

#include "flexflow/optimizer.h"
#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

Optimizer::Optimizer(FFModel const *_model) : model(_model) {}

ParallelTensor create_replica_parameter(FFModel const *model,
                                        const ParallelTensor p) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  ParallelTensor v = new ParallelTensorBase(*p);
  v->region_grad = LogicalRegion::NO_REGION;
  v->part_grad = LogicalPartition::NO_PART;
  v->region = runtime->create_logical_region(
      ctx, p->region.get_index_space(), p->region.get_field_space());
  if (v->sync_type == ParameterSyncType::PS) {
    // Do nothing
  } else if (v->sync_type == ParameterSyncType::NCCL) {
    v->part = runtime->get_logical_partition(
        ctx, v->region, p->part.get_index_partition());
  } else {
    assert(false);
  }
  return v;
}

SGDOptimizer::SGDOptimizer(FFModel const *_model,
                           double _lr,
                           double _momentum,
                           bool _nesterov,
                           double _weight_decay)
    : Optimizer(_model), lr(_lr), momentum(_momentum), nesterov(_nesterov),
      weight_decay(_weight_decay) {}

void SGDOptimizer::init(void) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  Initializer *initializer = new ZeroInitializer();
  for (size_t i = 0; i < model->parameters.size(); i++) {
    ParallelTensor p = model->parameters[i];
    Domain domain =
        runtime->get_index_space_domain(ctx, p->region.get_index_space());
    switch (domain.get_dim()) {
      case 0: {
        // Do not support 0-dim parameter
        assert(false);
        break;
      }
      case 1:
      case 2:
      case 3:
      case 4:
      case 5: {
        if (momentum > 0.0f) {
          v_values[p->region] = create_replica_parameter(model, p);
          initializer->init(model, v_values[p->region]);
        }
        break;
      }
      default: {
        // Unsupported dim
        assert(false);
        break;
      }
    }
  }
  delete initializer;
}

void SGDOptimizer::next(void) {}

void SGDOptimizer::update(const ParallelTensor p) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  assert(p->owner_op != NULL);
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(SGD_UPD_PS_TASK_ID,
                          TaskArgument(this, sizeof(SGDOptimizer)),
                          Predicate::TRUE_PRED,
                          0 /*mapper_id*/,
                          p->machine_view.hash());
    // regions[0]: region_grad
    launcher.add_region_requirement(RegionRequirement(
        p->region_grad, READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->region, READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    if (momentum > 0.0f) {
      // regions[2]: v_region
      assert(v_values.find(p->region) != v_values.end());
      launcher.add_region_requirement(
          RegionRequirement(v_values[p->region]->region,
                            READ_WRITE,
                            EXCLUSIVE,
                            v_values[p->region]->region));
      launcher.add_field(2, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
    // Parameter prefetching optimizations to reduce comm. overhead
    // Directly send the parameters back to all worker devices after SGD
    ArgumentMap argmap;
    IndexLauncher index_launcher(PS_PREFETCH_TASK_ID,
                                 p->parallel_is,
                                 TaskArgument(NULL, 0),
                                 argmap,
                                 Predicate::TRUE_PRED,
                                 false /*must*/,
                                 0 /*mapper_id*/,
                                 p->machine_view.hash());
    // regions[0]: region
    index_launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection*/, READ_ONLY, EXCLUSIVE, p->region));
    index_launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, index_launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    // Currently assume that we don't change the owner_op of weights
    // during fusion; thus the owner of a weight cannot be FusedOp
    assert(p->owner_op->op_type != OP_FUSED);
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    Domain domain = runtime->get_index_space_domain(ctx, p->parallel_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      OpMeta *mp = p->owner_op->meta[idx++];                                   \
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta *)));              \
    }                                                                          \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    IndexLauncher launcher(SGD_UPD_NCCL_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(SGDOptimizer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must_epoch*/,
                           0 /*mapper_id*/,
                           p->machine_view.hash());
    // regions[0]: region_grad
    launcher.add_region_requirement(RegionRequirement(p->part_grad,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    if (momentum > 0.0f) {
      // regions[2]: v_value
      assert(v_values.find(p->region) != v_values.end());
      launcher.add_region_requirement(
          RegionRequirement(v_values[p->region]->part,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            v_values[p->region]->region));
      launcher.add_field(2, FID_DATA);
    }
    // MustEpochLauncher must_epoch_launcher;
    // must_epoch_launcher.add_index_task(launcher);
    launcher.concurrent = true;
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    // runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
  } else {
    assert(false);
  }
}

void SGDOptimizer::ps_update_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  SGDOptimizer const *op = (SGDOptimizer *)task->args;
  if (op->momentum > 0.0f) {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
  }
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  float const *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL;
  size_t size = 0, num_replicas = 0;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<float, DIM> accWGrad(                                      \
        regions[0], task->regions[0], FID_DATA, ctx, runtime);                 \
    TensorAccessorW<float, DIM> accW(regions[1],                               \
                                     task->regions[1],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    for (int i = 0; i < domain.get_dim() - 1; i++) {                           \
      assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);                          \
      assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);                          \
    }                                                                          \
    size = accW.rect.volume();                                                 \
    assert(accWGrad.rect.volume() % accW.rect.volume() == 0);                  \
    num_replicas = accWGrad.rect.volume() / accW.rect.volume();                \
    w_grad_ptr = accWGrad.ptr;                                                 \
    w_ptr = accW.ptr;                                                          \
    if (op->momentum > 0.0f) {                                                 \
      TensorAccessorW<float, DIM> accV(regions[2],                             \
                                       task->regions[2],                       \
                                       FID_DATA,                               \
                                       ctx,                                    \
                                       runtime,                                \
                                       true /*readOutput*/);                   \
      assert(accW.rect == accV.rect);                                          \
      v_ptr = accV.ptr;                                                        \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dims
      assert(false);
    }
  }

  ps_update_task_gpu(op, w_grad_ptr, size, num_replicas, w_ptr, v_ptr);
}

#ifdef FF_USE_NCCL
void SGDOptimizer::nccl_update_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  SGDOptimizer const *op = (SGDOptimizer *)task->args;
  OpMeta const *meta = *((OpMeta **)task->local_args);
  // FFHandler handler = *((FFHandler*) task->local_args);
  if (op->momentum > 0.0f) {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
  }
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  float const *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL;
  size_t size = 0;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<float, DIM> accWGrad(                                      \
        regions[0], task->regions[0], FID_DATA, ctx, runtime);                 \
    TensorAccessorW<float, DIM> accW(regions[1],                               \
                                     task->regions[1],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    assert(accW.rect == accWGrad.rect);                                        \
    size = accW.rect.volume();                                                 \
    w_grad_ptr = accWGrad.ptr;                                                 \
    w_ptr = accW.ptr;                                                          \
    if (op->momentum > 0.0f) {                                                 \
      TensorAccessorW<float, DIM> accV(regions[2],                             \
                                       task->regions[2],                       \
                                       FID_DATA,                               \
                                       ctx,                                    \
                                       runtime,                                \
                                       true /*readOutput*/);                   \
      assert(accW.rect == accV.rect);                                          \
      v_ptr = accV.ptr;                                                        \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dims
      assert(false);
    }
  }

  nccl_update_task_gpu(op, meta, w_grad_ptr, size, w_ptr, v_ptr);
}
#endif

// ------------------------------------------------------------------
//                        Adam Optimizer
// ------------------------------------------------------------------

AdamOptimizer::AdamOptimizer(FFModel const *_model,
                             double _alpha,
                             double _beta1,
                             double _beta2,
                             double _weight_decay,
                             double _epsilon)
    : Optimizer(_model), alpha(_alpha), beta1(_beta1), beta2(_beta2),
      weight_decay(_weight_decay), epsilon(_epsilon), alpha_t(_alpha),
      beta1_t(1.0f), beta2_t(1.0f) {}

void AdamOptimizer::init(void) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  Initializer *initializer = new ZeroInitializer();
  reservedWorkSpaceSize = 0;
  for (size_t i = 0; i < model->parameters.size(); i++) {
    ParallelTensor p = model->parameters[i];
    Domain domain =
        runtime->get_index_space_domain(ctx, p->region.get_index_space());
    switch (domain.get_dim()) {
      case 0: {
        // Do not support 0-dim parameter
        assert(false);
        break;
      }
      case 1:
      case 2:
      case 3:
      case 4:
      case 5: {
        v_values[p->region] = create_replica_parameter(model, p);
        m_values[p->region] = create_replica_parameter(model, p);
        initializer->init(model, v_values[p->region]);
        initializer->init(model, m_values[p->region]);
        break;
      }
      default: {
        // Unsupported dim
        assert(false);
        break;
      }
    }
  }
  delete initializer;
}

void AdamOptimizer::set_weight_decay(double _weight_decay) {
  weight_decay = _weight_decay;
}

void AdamOptimizer::next(void) {
  beta1_t *= beta1;
  beta2_t *= beta2;
  alpha_t = alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
  // fprintf(stderr, "lr = %.4lf alpha_t = %.4lf\n", alpha, alpha_t);
}

void AdamOptimizer::update(const ParallelTensor p) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  assert(v_values.find(p->region) != v_values.end());
  assert(m_values.find(p->region) != m_values.end());
  assert(p->owner_op != NULL);
  reservedWorkSpaceSize += p->get_volume() * sizeof(float);
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(ADAM_UPD_PS_TASK_ID,
                          TaskArgument(this, sizeof(AdamOptimizer)),
                          Predicate::TRUE_PRED,
                          0 /*mapper_id*/,
                          p->machine_view.hash());
    // regions[0]: region_grad
    launcher.add_region_requirement(RegionRequirement(
        p->region_grad, READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->region, READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    // regions[2]: w_region
    launcher.add_region_requirement(
        RegionRequirement(v_values[p->region]->region,
                          READ_WRITE,
                          EXCLUSIVE,
                          v_values[p->region]->region));
    launcher.add_field(2, FID_DATA);
    // regions[3]: m_region
    launcher.add_region_requirement(
        RegionRequirement(m_values[p->region]->region,
                          READ_WRITE,
                          EXCLUSIVE,
                          m_values[p->region]->region));
    launcher.add_field(3, FID_DATA);
    runtime->execute_task(ctx, launcher);
    // Parameter prefetching optimizations to reduce comm. overhead
    // Directly send the parameters back to all worker devices after SGD
    ArgumentMap argmap;
    IndexLauncher index_launcher(PS_PREFETCH_TASK_ID,
                                 p->parallel_is,
                                 TaskArgument(NULL, 0),
                                 argmap,
                                 Predicate::TRUE_PRED,
                                 false /*must*/,
                                 0 /*mapper_id*/,
                                 p->machine_view.hash());
    // regions[0]: region
    index_launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection*/, READ_ONLY, EXCLUSIVE, p->region));
    index_launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, index_launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    assert(p->parallel_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    Domain domain = runtime->get_index_space_domain(ctx, p->parallel_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      OpMeta *mp = p->owner_op->meta[idx++];                                   \
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta *)));              \
    }                                                                          \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    IndexLauncher launcher(ADAM_UPD_NCCL_TASK_ID,
                           p->parallel_is,
                           TaskArgument(this, sizeof(AdamOptimizer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must_epoch*/,
                           0 /*mapper_id*/,
                           p->machine_view.hash());
    // regions[0]: region_grad
    launcher.add_region_requirement(RegionRequirement(p->part_grad,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(RegionRequirement(
        p->part, 0 /*projection id*/, READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    // regions[2]: w_region
    launcher.add_region_requirement(
        RegionRequirement(v_values[p->region]->part,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          v_values[p->region]->region));
    launcher.add_field(2, FID_DATA);
    // regions[3]: m_region
    launcher.add_region_requirement(
        RegionRequirement(m_values[p->region]->part,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          m_values[p->region]->region));
    launcher.add_field(3, FID_DATA);
    // MustEpochLauncher must_epoch_launcher;
    // must_epoch_launcher.add_index_task(launcher);
    launcher.concurrent = true;
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    // runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
  } else {
    assert(false);
  }
}

void SGDOptimizer::unified_update(std::vector<ParallelTensor> const parameters) {
  //todo
}

void AdamOptimizer::unified_update(std::vector<ParallelTensor> const parameters) {
  Context ctx = model->config.lg_ctx;
  Runtime *runtime = model->config.lg_hlr;
  const ParallelTensor p0 = parameters.at(0);
  ArgumentMap argmap;
    Domain domain = runtime->get_index_space_domain(ctx, p0->parallel_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      OpMeta *mp = p0->owner_op->meta[idx++];                                   \
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta *)));              \
    }                                                                          \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }

  int offset = 0;
  int processed_parameters_num = 0;
  // printf("param size: %d\n", parameters.size());

  size_t workSpaceSize = model->handlers->workSpaceSize *
                         model->config.workersPerNode * model->config.numNodes;

  while (processed_parameters_num < parameters.size()) {
    parameters_num = 0;

    for(int i = processed_parameters_num; i < parameters.size(); i++){
      const ParallelTensor p = parameters.at(i);
      assert(v_values.find(p->region) != v_values.end());
      assert(m_values.find(p->region) != m_values.end());
      assert(p->owner_op != NULL);
      if (reservedWorkSpaceSize + p->get_volume() * sizeof(float) >= workSpaceSize) {
        break;
      }
      reservedWorkSpaceSize += p->get_volume() * sizeof(float);
      parameters_num += 1;
      assert(p->sync_type == ParameterSyncType::NCCL);
      assert(p->parallel_is != IndexSpace::NO_SPACE);
    }

    // printf("parameters_num: %d %zu, %zu, %d\n", parameters_num,
    // reservedWorkSpaceSize, model->handlers->workSpaceSize,
    // parameters.size());
    assert(processed_parameters_num <= parameters.size());

    IndexLauncher launcher(ADAM_UNIFY_UPD_NCCL_TASK_ID,
                           p0->parallel_is,
                           TaskArgument(this, sizeof(AdamOptimizer)),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must_epoch*/,
                           0 /*mapper_id*/,
                           p0->machine_view.hash());
    // launch a unified task
    for (int j = 0; j < parameters_num; j++) {
        const ParallelTensor p = parameters.at(processed_parameters_num + j);

      // regions[0]: region_grad
      launcher.add_region_requirement(RegionRequirement(p->part_grad,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        p->region_grad));
      launcher.add_field(offset, FID_DATA);
      // regions[1]: region
      launcher.add_region_requirement(RegionRequirement(
          p->part, 0 /*projection id*/, READ_WRITE, EXCLUSIVE, p->region));
      launcher.add_field(offset + 1, FID_DATA);
      // regions[2]: w_region
      launcher.add_region_requirement(
          RegionRequirement(v_values[p->region]->part,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            v_values[p->region]->region));
      launcher.add_field(offset + 2, FID_DATA);
      // regions[3]: m_region
      launcher.add_region_requirement(
          RegionRequirement(m_values[p->region]->part,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            m_values[p->region]->region));
      launcher.add_field(offset + 3, FID_DATA);
      offset += 4;
    }

    // update alpha, beta
    for (int i = 0; i < parameters_num; i++) {
      this->next();
    }
    launcher.concurrent = true;
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    // runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
    reservedWorkSpaceSize = 0;
    offset = 0;
    processed_parameters_num += parameters_num;
  }
  parameters_num = 0;
}

void AdamOptimizer::ps_update_task(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  AdamOptimizer const *op = (AdamOptimizer *)task->args;
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  float const *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL, *m_ptr = NULL;
  size_t size = 0, num_replicas = 0;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<float, DIM> accWGrad(                                      \
        regions[0], task->regions[0], FID_DATA, ctx, runtime);                 \
    TensorAccessorW<float, DIM> accW(regions[1],                               \
                                     task->regions[1],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    TensorAccessorW<float, DIM> accV(regions[2],                               \
                                     task->regions[2],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    TensorAccessorW<float, DIM> accM(regions[3],                               \
                                     task->regions[3],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    size = accW.rect.volume();                                                 \
    assert(accWGrad.rect.volume() % accW.rect.volume() == 0);                  \
    num_replicas = accWGrad.rect.volume() / accW.rect.volume();                \
    w_grad_ptr = accWGrad.ptr;                                                 \
    w_ptr = accW.ptr;                                                          \
    v_ptr = accV.ptr;                                                          \
    m_ptr = accM.ptr;                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dims
      assert(false);
    }
  }

  ps_update_task_gpu(op, w_grad_ptr, size, num_replicas, w_ptr, v_ptr, m_ptr);
}

#ifdef FF_USE_NCCL
void AdamOptimizer::nccl_update_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  AdamOptimizer const *op = (AdamOptimizer *)task->args;
  OpMeta const *meta = *((OpMeta **)task->local_args);
  // FFHandler handler = *((FFHandler*) task->local_args);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  float const *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL, *m_ptr = NULL;
  size_t size = 0;
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<float, DIM> accWGrad(                                      \
        regions[0], task->regions[0], FID_DATA, ctx, runtime);                 \
    TensorAccessorW<float, DIM> accW(regions[1],                               \
                                     task->regions[1],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    TensorAccessorW<float, DIM> accV(regions[2],                               \
                                     task->regions[2],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    TensorAccessorW<float, DIM> accM(regions[3],                               \
                                     task->regions[3],                         \
                                     FID_DATA,                                 \
                                     ctx,                                      \
                                     runtime,                                  \
                                     true /*readOutput*/);                     \
    size = accW.rect.volume();                                                 \
    assert(accWGrad.rect == accW.rect);                                        \
    assert(accWGrad.rect == accV.rect);                                        \
    assert(accWGrad.rect == accM.rect);                                        \
    w_grad_ptr = accWGrad.ptr;                                                 \
    w_ptr = accW.ptr;                                                          \
    v_ptr = accV.ptr;                                                          \
    m_ptr = accM.ptr;                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dims
      assert(false);
    }
  }

  nccl_update_task_gpu(op, meta, w_grad_ptr, size, w_ptr, v_ptr, m_ptr);
}

void AdamOptimizer::nccl_unified_update_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  // assert(regions.size() == 4);
  // assert(task->regions.size() == 4);
  AdamOptimizer const *op = (AdamOptimizer *)task->args;
  OpMeta const *meta = *((OpMeta **)task->local_args);
  // FFHandler handler = *((FFHandler*) task->local_args);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  // float const *w_grad_ptr[op->parameters_num];
  // float *w_ptr[op->parameters_num], *v_ptr[op->parameters_num],
  // *m_ptr[op->parameters_num];

  // hipMalloc(w_grad_ptr, sizeof(float*) * op->parameters_num);
  // hipMalloc(w_ptr, sizeof(float*) * op->parameters_num);
  // hipMalloc(v_ptr, sizeof(float*) * op->parameters_num);
  // hipMalloc(m_ptr, sizeof(float*) * op->parameters_num);
  GenericTensorAccessorR accWGrads[op->parameters_num];
  GenericTensorAccessorW accWs[op->parameters_num];
  GenericTensorAccessorW accVs[op->parameters_num];
  GenericTensorAccessorW accMs[op->parameters_num];
  size_t *size = new size_t[op->parameters_num];
  int offset = 0;

  // printf("parameters_num: %d\n", op->parameters_num);

  for (int i = 0; i < op->parameters_num; i++) {
    accWGrads[i] = helperGetGenericTensorAccessorRO(DataType::DT_FLOAT,
                                                    regions[offset],
                                                    task->regions[offset],
                                                    FID_DATA,
                                                    ctx,
                                                    runtime);
    accWs[i] = helperGetGenericTensorAccessorWO(DataType::DT_FLOAT,
                                                regions[offset + 1],
                                                task->regions[offset + 1],
                                                FID_DATA,
                                                ctx,
                                                runtime);
    accVs[i] = helperGetGenericTensorAccessorWO(DataType::DT_FLOAT,
                                                regions[offset + 2],
                                                task->regions[offset + 2],
                                                FID_DATA,
                                                ctx,
                                                runtime);
    accMs[i] = helperGetGenericTensorAccessorWO(DataType::DT_FLOAT,
                                                regions[offset + 3],
                                                task->regions[offset + 3],
                                                FID_DATA,
                                                ctx,
                                                runtime);
    offset += 4;

    size[i] = accWGrads[i].domain.get_volume();
    // w_grad_ptr[i] = accWGrad.get_float_ptr();
    // w_ptr[i] = accW.get_float_ptr();
    // v_ptr[i] = accV.get_float_ptr();
    // m_ptr[i] = accM.get_float_ptr();
  }
  nccl_unified_update_task_gpu(op, meta, accWGrads, size, accWs, accVs, accMs);
}
#endif

}; // namespace FlexFlow
