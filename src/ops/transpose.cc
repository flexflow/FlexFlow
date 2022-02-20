/* Copyright 2020 Facebook
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

#include "flexflow/ops/transpose.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::coord_t;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::transpose(const Tensor input,
                          const std::vector<int>& perm,
                          const char* name)
{
  assert(false);
#ifdef DEADCODE
  Transpose* transpose = new Transpose(*this, input, perm, name);
  layers.push_back(transpose);
  return transpose->outputs[0];
#endif
}

Transpose::Transpose(FFModel& model,
                     const ParallelTensor input,
                     const std::vector<int>& _perm,
                     const char* name)
: Op(model, OP_TRANSPOSE, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, input)
{
  assert(_perm.size() == input->num_dims);
  // Use Legion indexing to store perm
  for (int i = 0; i < input->num_dims; i++)
    perm[i] = input->num_dims - 1 - _perm[input->num_dims - 1 - i];
  ParallelDim dims[MAX_TENSOR_DIM];
  int numdim = input->num_dims;
  for (int i = 0; i < numdim; i++)
    dims[i] = input->dims[perm[i]];
  outputs[0] = model.create_parallel_tensor_legion_ordering(numdim, dims, input->data_type, this);
}

void Transpose::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(TRANSPOSE_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Transpose)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Transpose::init_meta(TransposeMeta *m, Domain const &in_domain, Domain const &out_domain) const
{
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[this->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[this->perm[i]]);
  }
  m->num_dim = out_domain.get_dim();
  for (int i = 0; i < m->num_dim; i++)
    m->perm[i] = this->perm[i];
}

OpMeta* Transpose::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* transpose = (const Transpose*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());

  TransposeMeta* m = new TransposeMeta(handle);
  transpose->init_meta(m, in_domain, out_domain);
  m->profiling = transpose->profiling;
  return m;
}

void Transpose::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(TRANSPOSE_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, false), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Transpose::forward_task(const Task* task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[m->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[m->perm[i]]);
  }
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Transpose::forward_kernel_wrapper(m, in_ptr, out_ptr, in_domain, out_domain);
}

void Transpose::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(TRANSPOSE_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Transpose::backward_task(const Task* task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_grad_domain.get_dim(); i++) {
    assert(out_grad_domain.hi()[i] == in_grad_domain.hi()[m->perm[i]]);
    assert(out_grad_domain.lo()[i] == in_grad_domain.lo()[m->perm[i]]);
  }
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Transpose::backward_kernel_wrapper(m, in_grad_ptr, out_grad_ptr, in_grad_domain, out_grad_domain);
}

bool Transpose::measure_operator_cost(
    Simulator* sim,
    const MachineView& mv,
    CostMetrics& cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }

  TransposeMeta *m = sim->transpose_meta;
  this->init_meta(m, sub_input.get_domain(), sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, input_ptr, output_ptr, sub_input.get_domain(), sub_output.get_domain());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel_wrapper(m, input_grad_ptr, output_grad_ptr, sub_input.get_domain(), sub_output.get_domain());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow
