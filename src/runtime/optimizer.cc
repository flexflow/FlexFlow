/* Copyright 2020 Facebook, Stanford
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

#include "optimizer.h"
#include "model.h"

Optimizer::Optimizer(const FFModel* _model)
: model(_model) {}

Parameter create_replica_parameter(const FFModel* model,
                                   const Parameter& p)
{
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  Parameter v;
  v.sync_type = p.sync_type;
  v.owner_op = p.owner_op;
  v.region = runtime->create_logical_region(
      ctx, p.region.get_index_space(), p.region.get_field_space());
  if (v.sync_type == ParameterSyncType::PS) {
    // Do nothing
  } else if (v.sync_type == ParameterSyncType::NCCL) {
    v.part = runtime->get_logical_partition(
        ctx, v.region, p.part.get_index_partition());
  } else {
    assert(false);
  }
  return v;
}

SGDOptimizer::SGDOptimizer(const FFModel* _model,
                           double _lr, double _momentum,
                           bool _nesterov, double _weight_decay)
: Optimizer(_model), lr(_lr), momentum(_momentum),
  nesterov(_nesterov), weight_decay(_weight_decay)
{}

void SGDOptimizer::init(void)
{
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  Initializer* initializer = new ZeroInitializer();
  for (size_t i = 0; i < model->parameters.size(); i++) {
    Parameter p = model->parameters[i];
    Domain domain = runtime->get_index_space_domain(
        ctx, p.region.get_index_space());
    switch (domain.get_dim()) {
      case 0:
      {
        // Do not support 0-dim parameter
        assert(false);
        break;
      }
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      {
        if (momentum > 0.0f) {
          v_values[p.region] = create_replica_parameter(model, p);
          initializer->init(model, &v_values[p.region]);
        }
        break;
      }
      default:
      {
        // Unsupported dim
        assert(false);
        break;
      }
    }
  }
  delete initializer;
}

void SGDOptimizer::next(void)
{
}

void SGDOptimizer::update(const Parameter* p)
{
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  assert(p->owner_op != NULL);
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(SGD_UPD_PS_TASK_ID,
        TaskArgument(this, sizeof(SGDOptimizer)),
        Predicate::TRUE_PRED, 0/*mapper_id*/,
        FFConfig::get_hash_id(std::string(p->owner_op->name)));
    // regions[0]: region_grad
    launcher.add_region_requirement(
        RegionRequirement(p->region_grad,
                          READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->region,
                          READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    if (momentum > 0.0f) {
      // regions[2]: v_region
      assert(v_values.find(p->region) != v_values.end());
      launcher.add_region_requirement(
          RegionRequirement(v_values[p->region].region,
                            READ_WRITE, EXCLUSIVE, v_values[p->region].region));
      launcher.add_field(2, FID_DATA);
    }
    runtime->execute_task(ctx, launcher);
    // Parameter prefetching optimizations to reduce comm. overhead
    // Directly send the parameters back to all worker devices after SGD
    ArgumentMap argmap;
    IndexLauncher index_launcher(PS_PREFETCH_TASK_ID, p->owner_op->task_is,
        TaskArgument(NULL, 0), argmap,
        Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
        FFConfig::get_hash_id(std::string(p->owner_op->name)));
    // regions[0]: region
    index_launcher.add_region_requirement(
        RegionRequirement(p->part, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, p->region));
    index_launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, index_launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    IndexSpace task_is = p->owner_op->task_is;
    assert(task_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    Domain domain = runtime->get_index_space_domain(ctx, task_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
      case DIM: \
      { \
        Rect<DIM> rect = domain; \
        ParallelConfig pc; \
        model->config.find_parallel_config(DIM, p->owner_op->name, pc); \
        int idx = 0; \
        for (PointInRectIterator<DIM> it(rect); it(); it++) { \
          OpMeta* mp = p->owner_op->meta[idx++]; \
          argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
        } \
        break; \
      }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    IndexLauncher launcher(SGD_UPD_NCCL_TASK_ID, task_is,
        TaskArgument(this, sizeof(SGDOptimizer)), argmap,
        Predicate::TRUE_PRED, false/*must_epoch*/, 0/*mapper_id*/,
        FFConfig::get_hash_id(p->owner_op->name));
    // regions[0]: region_grad
    launcher.add_region_requirement(
        RegionRequirement(p->part_grad, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->part, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    if (momentum > 0.0f) {
      // regions[2]: v_value
      assert(v_values.find(p->region) != v_values.end());
      launcher.add_region_requirement(
          RegionRequirement(v_values[p->region].part, 0/*projection id*/,
                            READ_WRITE, EXCLUSIVE, v_values[p->region].region));
      launcher.add_field(2, FID_DATA);
    }
    //MustEpochLauncher must_epoch_launcher;
    //must_epoch_launcher.add_index_task(launcher);
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    //runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
  } else {
    assert(false);
  }
}

// ------------------------------------------------------------------
//                        Adam Optimizer
// ------------------------------------------------------------------

AdamOptimizer::AdamOptimizer(const FFModel* _model,
                             double _alpha, double _beta1,
                             double _beta2, double _weight_decay,
                             double _epsilon)
: Optimizer(_model), alpha(_alpha), beta1(_beta1), beta2(_beta2),
  weight_decay(_weight_decay),
  epsilon(_epsilon), alpha_t(_alpha), beta1_t(1.0f), beta2_t(1.0f)
{}

void AdamOptimizer::init(void)
{
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  Initializer* initializer = new ZeroInitializer();
  for (size_t i = 0; i < model->parameters.size(); i++) {
    Parameter p = model->parameters[i];
    Domain domain = runtime->get_index_space_domain(
        ctx, p.region.get_index_space());
    switch (domain.get_dim()) {
      case 0:
      {
        // Do not support 0-dim parameter
        assert(false);
        break;
      }
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      {
        v_values[p.region] = create_replica_parameter(model, p);
        m_values[p.region] = create_replica_parameter(model, p);
        initializer->init(model, &v_values[p.region]);
        initializer->init(model, &m_values[p.region]);
        break;
      }
      default:
      {
        // Unsupported dim
        assert(false);
        break;
      }
    }
  }
  delete initializer;
}

void AdamOptimizer::set_weight_decay(double _weight_decay)
{
  weight_decay = _weight_decay;
}

void AdamOptimizer::next(void)
{
  beta1_t *= beta1;
  beta2_t *= beta2;
  alpha_t = alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
  //fprintf(stderr, "lr = %.4lf alpha_t = %.4lf\n", alpha, alpha_t);
}

void AdamOptimizer::update(const Parameter* p)
{
  Context ctx = model->config.lg_ctx;
  Runtime* runtime = model->config.lg_hlr;
  assert(v_values.find(p->region) != v_values.end());
  assert(m_values.find(p->region) != m_values.end());
  assert(p->owner_op != NULL);
  if (p->sync_type == ParameterSyncType::PS) {
    TaskLauncher launcher(ADAM_UPD_PS_TASK_ID,
        TaskArgument(this, sizeof(AdamOptimizer)),
        Predicate::TRUE_PRED, 0/*mapper_id*/,
        FFConfig::get_hash_id(std::string(p->owner_op->name)));
    // regions[0]: region_grad
    launcher.add_region_requirement(
        RegionRequirement(p->region_grad,
                          READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->region,
                          READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    // regions[2]: w_region
    launcher.add_region_requirement(
        RegionRequirement(v_values[p->region].region,
                          READ_WRITE, EXCLUSIVE, v_values[p->region].region));
    launcher.add_field(2, FID_DATA);
    // regions[3]: m_region
    launcher.add_region_requirement(
        RegionRequirement(m_values[p->region].region,
                          READ_WRITE, EXCLUSIVE, m_values[p->region].region));
    launcher.add_field(3, FID_DATA);
    runtime->execute_task(ctx, launcher);
    // Parameter prefetching optimizations to reduce comm. overhead
    // Directly send the parameters back to all worker devices after SGD
    ArgumentMap argmap;
    IndexLauncher index_launcher(PS_PREFETCH_TASK_ID, p->owner_op->task_is,
        TaskArgument(NULL, 0), argmap,
        Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
        FFConfig::get_hash_id(std::string(p->owner_op->name)));
    // regions[0]: region
    index_launcher.add_region_requirement(
        RegionRequirement(p->part, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, p->region));
    index_launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, index_launcher);
  } else if (p->sync_type == ParameterSyncType::NCCL) {
    IndexSpace task_is = p->owner_op->task_is;
    assert(task_is != IndexSpace::NO_SPACE);
    ArgumentMap argmap;
    Domain domain = runtime->get_index_space_domain(ctx, task_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
      case DIM: \
      { \
        Rect<DIM> rect = domain; \
        ParallelConfig pc; \
        model->config.find_parallel_config(DIM, p->owner_op->name, pc); \
        int idx = 0; \
        for (PointInRectIterator<DIM> it(rect); it(); it++) { \
          OpMeta* mp = p->owner_op->meta[idx++]; \
          argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
        } \
        break; \
      }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
    IndexLauncher launcher(ADAM_UPD_NCCL_TASK_ID, task_is,
        TaskArgument(this, sizeof(AdamOptimizer)), argmap,
        Predicate::TRUE_PRED, false/*must_epoch*/, 0/*mapper_id*/,
        FFConfig::get_hash_id(p->owner_op->name));
    // regions[0]: region_grad
    launcher.add_region_requirement(
        RegionRequirement(p->part_grad, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, p->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1]: region
    launcher.add_region_requirement(
        RegionRequirement(p->part, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, p->region));
    launcher.add_field(1, FID_DATA);
    // regions[2]: w_region
    launcher.add_region_requirement(
        RegionRequirement(v_values[p->region].part, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, v_values[p->region].region));
    launcher.add_field(2, FID_DATA);
    // regions[3]: m_region
    launcher.add_region_requirement(
        RegionRequirement(m_values[p->region].part, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, m_values[p->region].region));
    launcher.add_field(3, FID_DATA);
    //MustEpochLauncher must_epoch_launcher;
    //must_epoch_launcher.add_index_task(launcher);
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    //runtime->execute_must_epoch(ctx, must_epoch_launcher);
    runtime->issue_execution_fence(ctx);
  } else {
    assert(false);
  }
}
