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

#include "optimizer.h"
#include "model.h"

Optimizer::Optimizer(const FFModel* _model)
: model(_model) {}

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
    Tensor p = model->parameters[i];
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
      {
        if (momentum > 0.0f) {
          v_regions[p.region] = runtime->create_logical_region(
              ctx, p.region.get_index_space(), p.region.get_field_space());
          Tensor t;
          t.region = v_regions[p.region];
          initializer->init(ctx, runtime, &t);
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
  TaskLauncher launcher(SGD_UPD_TASK_ID,
                        TaskArgument(this, sizeof(SGDOptimizer)),
                        Predicate::TRUE_PRED, 0/*mapper_id*/,
                        FFConfig::get_hash_id(std::string(p->pcname)));
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
    assert(v_regions.find(p->region) != v_regions.end());
    launcher.add_region_requirement(
        RegionRequirement(v_regions[p->region],
                          READ_WRITE, EXCLUSIVE, v_regions[p->region]));
    launcher.add_field(2, FID_DATA);
  }
  runtime->execute_task(ctx, launcher);
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
    Tensor p = model->parameters[i];
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
      {
        v_regions[p.region] = runtime->create_logical_region(
            ctx, p.region.get_index_space(), p.region.get_field_space());
        m_regions[p.region] = runtime->create_logical_region(
            ctx, p.region.get_index_space(), p.region.get_field_space());
        Tensor t;
        // Zeros v_regions and m_regions
        t.region = v_regions[p.region];
        initializer->init(ctx, runtime, &t);
        t.region = m_regions[p.region];
        initializer->init(ctx, runtime, &t);
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
  assert(v_regions.find(p->region) != v_regions.end());
  assert(m_regions.find(p->region) != m_regions.end());
  TaskLauncher launcher(ADAM_UPD_TASK_ID,
                        TaskArgument(this, sizeof(AdamOptimizer)),
                        Predicate::TRUE_PRED, 0/*mapper_id*/,
                        FFConfig::get_hash_id(std::string(p->pcname)));
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
      RegionRequirement(v_regions[p->region],
                        READ_WRITE, EXCLUSIVE, v_regions[p->region]));
  launcher.add_field(2, FID_DATA);
  // regions[3]: m_region
  launcher.add_region_requirement(
      RegionRequirement(m_regions[p->region],
                        READ_WRITE, EXCLUSIVE, m_regions[p->region]));
  launcher.add_field(3, FID_DATA);
  runtime->execute_task(ctx, launcher);
}
