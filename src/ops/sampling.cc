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

#include "flexflow/ops/sampling.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif

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
using PCG::Node;

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]
Tensor FFModel::sampling(const Tensor input, float top_p, char const *name) {
  Layer *li = new Layer(this,
                        OP_SAMPLING,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        1 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    // now just support 1 output
    dims[0] = 1;
    // li->outputs[0] = create_tensor_legion_ordering(
    //     numdims, dims, input->data_type, li, 0, true /*create_grad*/);
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, false /*create_grad*/);
  }
  layers.push_back(li);
  li->add_float_property("top_p", top_p);
  // outputs[0] = li->outputs[0];
  // outputs[1] = li->outputs[1];
  return li->outputs[0];
}

Op *Sampling::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float top_p;
  layer->get_float_property("top_p", top_p);
  return new Sampling(model, inputs[0], top_p, layer->name);
}

SamplingParams Sampling::get_params() const {
  SamplingParams params;
  params.top_p = this->top_p;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool SamplingParams::is_valid(ParallelTensorShape const &) const {
  return true;
}

bool operator==(SamplingParams const &lhs, SamplingParams const &rhs) {
  return lhs.top_p == rhs.top_p;
}

Sampling::Sampling(FFModel &model,
                   const ParallelTensor _input,
                   float _top_p,
                   char const *name)
    : Op(model,
         OP_SAMPLING,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input),
      top_p(_top_p) {
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  dims[0].size = 1;
  std::cout << "degree: " << inputs[0]->dims[0].degree << "\n";
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  //   outputs[0] = model.create_parallel_tensor_legion_ordering(
  //       numdim, dims, _input->data_type, this, 0 /*owner_idx*/);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_INT32, this, 0 /*owner_idx*/);
}

Sampling::Sampling(FFModel &model,
                   Sampling const &other,
                   const ParallelTensor input)
    : Sampling(model, input, other.top_p, other.name) {}

Sampling::Sampling(FFModel &model,
                   SamplingParams const &params,
                   const ParallelTensor input,
                   char const *name)
    : Sampling(model, input, params.top_p, params.name) {}

void Sampling::init_inference(FFModel const &ff,
                              std::vector<ParallelTensor> const &batch_inputs,
                              std::vector<ParallelTensor> const &batch_outputs,
                              MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(SAMPLING_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Sampling)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void Sampling::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(SAMPLING_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Sampling)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Sampling::init_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  Sampling *s = (Sampling *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  GenericTensorAccessorW acc_input =
      helperGetGenericTensorAccessorRW(s->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);

  int length = acc_input.domain.hi()[0] - acc_input.domain.lo()[0] + 1;
  int batch_size = acc_input.domain.get_volume() / length;
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  SamplingMeta *m = new SamplingMeta(
      handle, s, batch_size, length * batch_size, acc_input, gpu_mem_allocator);
  m->profiling = s->profiling;
  m->inference_debugging = s->inference_debugging;
  std::strcpy(m->op_name, s->name);
  m->layer_guid = s->layer_guid;
  m->top_p = s->top_p;
  return m;
}

void Sampling::forward(FFModel const &ff) {
  // Sampling does not support forward
  assert(false);
}

FutureMap Sampling::inference(FFModel const &ff,
                              BatchConfigFuture const &bc,
                              std::vector<ParallelTensor> const &batch_inputs,
                              std::vector<ParallelTensor> const &batch_outputs,
                              MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  /* std::cout << "Sampling op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(SAMPLING_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

InferenceResult
    Sampling::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  // BatchConfig const *bc = (BatchConfig *)task->args;
  SamplingMeta *m = *((SamplingMeta **)task->local_args);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }

  GenericTensorAccessorW input = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);

  int batch_size = bc->num_active_tokens();
  Sampling::forward_kernel_wrapper(m, input, indices, batch_size);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    Sampling::save_inference_tensors_to_file(
        m, shard_id, bc, {}, {}, {input, indices});
  }

  InferenceResult ir;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size);
  return ir;
}

void Sampling::backward(FFModel const &ff) {
  // Sampling does not support backward
  assert(false);
}

void Sampling::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->top_p);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node Sampling::deserialize(FFModel &ff,
                           Legion::Deserializer &dez,
                           ParallelTensor inputs[],
                           int num_inputs) {
  assert(num_inputs == 1);
  float top_p;
  dez.deserialize(top_p);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  SamplingParams params;
  params.top_p = top_p;
  strcpy(params.name, name);
  return ff.get_or_create_node<Sampling>(inputs[0], params);
}

Op *Sampling::materialize(FFModel &ff,
                          ParallelTensor inputs[],
                          int num_inputs) const {
  SamplingParams params = get_params();
  return new Sampling(ff, params, inputs[0], this->name);
}

bool Sampling::measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SamplingParams>::operator()(
    FlexFlow::SamplingParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.top_p);
  return key;
}
}; // namespace std
