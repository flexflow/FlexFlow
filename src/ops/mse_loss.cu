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

LegionRuntime::Logger::Category log_mse("mse");

void FFModel::mse_loss(const std::string& pcname,
                       const Tensor& _logit,
                       const Tensor& _label,
                       const std::string& reduction)
{
  AggrMode aggr = AGGR_MODE_NONE;
  if (reduction == "sum")
    aggr = AGGR_MODE_SUM;
  else if (reduction == "average")
    aggr = AGGR_MODE_AVG;
  else
    assert(reduction == "none");
  MSELoss* op = new MSELoss(*this, pcname, _logit, _label, aggr);
  layers.push_back(op);
}

MSELoss::MSELoss(FFModel& model,
                 const std::string& pcname,
                 const Tensor& _logit,
                 const Tensor& _label,
                 AggrMode _aggr)
: Op(pcname, _logit, _label), profiling(model.config.profiling),
aggr_mode(_aggr)
{
  task_is = model.get_or_create_task_is(pcname);
  // Current assume 2D logit and label
  assert(_logit.numDim == 2);
  assert(_label.numDim == 2);
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is); 
}

void MSELoss::init(const FFModel& model)
{}

void MSELoss::forward(const FFModel& model)
{
}

__global__
void mseloss_backward(float* logitsGrad,
                      const float* logits,
                      const float* labels,
                      float factor,
                      int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    logitsGrad[i] = factor * (logits[i] - labels[i]);
  }
}

__host__
void MSELoss::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const MSELoss* op = (MSELoss*) task->args;
  TensorAccessorR<float, 2> accLogits(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> accLabels(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> accLogitsGrad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, false/*readOutput*/);
  assert(accLogits.rect == accLabels.rect);
  assert(accLogits.rect == accLogitsGrad.rect);
  float scale = 1.0f;
  switch (op->aggr_mode) {
    case AGGR_MODE_SUM:
      scale = 1.0f;
      break;
    case AGGR_MODE_AVG:
      // Divided by the global batch size
      scale = 1.0f / (op->inputs[0].adim[0]);
      break;
    default:
      assert(false);
  }
  mseloss_backward<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(
      accLogitsGrad.ptr, accLogits.ptr, accLabels.ptr,
      scale, accLogits.rect.volume());
  checkCUDA(cudaDeviceSynchronize());
}

void MSELoss::backward(const FFModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  IndexLauncher launcher(MSELOSS_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(MSELoss)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0]: _logit
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: _label
  launcher.add_region_requirement(
      RegionRequirement(inputs[1].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  // regions[2]: logit_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

