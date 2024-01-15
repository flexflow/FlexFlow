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

#include "flexflow/ops/argmax.h"
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

Tensor FFModel::argmax(const Tensor input, bool beam_search, char const *name) {
  Layer *li = new Layer(this,
                        OP_ARGMAX,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        beam_search ? 2 : 1 /*outputs*/,
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
    if (beam_search) {
      // parent id
      li->outputs[1] = create_tensor_legion_ordering(
          numdims, dims, DT_INT32, li, 1, false /*create_grad*/);
    }
  }
  li->add_int_property("beam_search", beam_search);
  layers.push_back(li);
  // outputs[0] = li->outputs[0];
  // outputs[1] = li->outputs[1];
  return li->outputs[0];
}

Op *ArgMax::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("beam_search", value);
  bool beam_search = (bool)value;
  return new ArgMax(model, inputs[0], beam_search, layer->name);
}

ArgMaxParams ArgMax::get_params() const {
  ArgMaxParams params;
  params.beam_search = this->beam_search;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool ArgMaxParams::is_valid(ParallelTensorShape const &) const {
  return true;
}

bool operator==(ArgMaxParams const &lhs, ArgMaxParams const &rhs) {
  return lhs.beam_search == rhs.beam_search;
}

ArgMax::ArgMax(FFModel &model,
               const ParallelTensor _input,
               bool _beam_search,
               char const *name)
    : Op(model,
         OP_ARGMAX,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         _beam_search ? 2 : 1 /*outputs*/,
         _input),
      beam_search(_beam_search) {
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  dims[0].size = 1;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  //   outputs[0] = model.create_parallel_tensor_legion_ordering(
  //       numdim, dims, _input->data_type, this, 0 /*owner_idx*/);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_INT32, this, 0 /*owner_idx*/);
  if (_beam_search) {
    outputs[1] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, DT_INT32, this, 1 /*owner_idx*/);
  }
}

ArgMax::ArgMax(FFModel &model, ArgMax const &other, const ParallelTensor input)
    : ArgMax(model, input, other.beam_search, other.name) {}

ArgMax::ArgMax(FFModel &model,
               ArgMaxParams const &params,
               const ParallelTensor input,
               char const *name)
    : ArgMax(model, input, params.beam_search, params.name) {}

void ArgMax::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(ARGMAX_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ArgMax)),
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

void ArgMax::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ARGMAX_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ArgMax)),
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

OpMeta *ArgMax::init_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  ArgMax *s = (ArgMax *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  GenericTensorAccessorW acc_input =
      helperGetGenericTensorAccessorRW(s->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  int length = acc_input.domain.hi()[0] - acc_input.domain.lo()[0] + 1;
  int batch_size = acc_input.domain.get_volume() / length;
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);

  ArgMaxMeta *m = new ArgMaxMeta(handle,
                                 s,
                                 input_domain,
                                 output_domain,
                                 acc_input,
                                 batch_size,
                                 length * batch_size,
                                 gpu_mem_allocator);
  m->profiling = s->profiling;
  m->inference_debugging = s->inference_debugging;
  m->beam_search = s->beam_search;
  std::strcpy(m->op_name, s->name);
  m->layer_guid = s->layer_guid;
  return m;
}

void ArgMax::forward(FFModel const &ff) {
  // ArgMax does not support forward
  assert(false);
}

FutureMap ArgMax::inference(FFModel const &ff,
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
  /* std::cout << "ArgMax op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  if (beam_search) {
    IndexLauncher launcher(ARGMAX_BEAM_INF_TASK_ID,
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
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[1]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[1]->region));
    launcher.add_field(2, FID_DATA);
    return runtime->execute_index_space(ctx, launcher);
  } else {
    IndexLauncher launcher(ARGMAX_NORM_INF_TASK_ID,
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
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(1, FID_DATA);
    return runtime->execute_index_space(ctx, launcher);
  }
}

BeamInferenceResult
    ArgMax::inference_task_beam(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    BeamInferenceResult ir;
    return ir;
  }
  ArgMaxMeta *m = *((ArgMaxMeta **)task->local_args);

  GenericTensorAccessorW input = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int batch_size = bc->num_active_tokens();
  GenericTensorAccessorW parent = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[2], task->regions[2], FID_DATA, ctx, runtime);
  ArgMax::forward_kernel_wrapper(m, input, indices, parent, batch_size);
  BeamInferenceResult ir;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size);
  download_tensor(m->probs, ir.probs, batch_size);
  download_tensor<int>(parent.get_int32_ptr(), ir.parent_id, batch_size);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    ArgMax::save_inference_tensors_to_file(
        m, shard_id, bc, {}, {}, {input, indices, parent});
  }

  return ir;
}

InferenceResult
    ArgMax::inference_task_norm(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  ArgMaxMeta *m = *((ArgMaxMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }

  GenericTensorAccessorW input = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW parent;
  int batch_size = bc->num_active_tokens();
  ArgMax::forward_kernel_wrapper(m, input, indices, parent, batch_size);
  InferenceResult ir;
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    ArgMax::save_inference_tensors_to_file(
        m, shard_id, bc, {}, {}, {input, indices});
  }

  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size);
  return ir;
}

void ArgMax::backward(FFModel const &ff) {
  // ArgMax does not support backward
  assert(false);
}

void ArgMax::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->beam_search);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node ArgMax::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 1);
  bool beam_search;
  dez.deserialize(beam_search);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  ArgMaxParams params;
  params.beam_search = beam_search;
  strcpy(params.name, name);
  return ff.get_or_create_node<ArgMax>(inputs[0], params);
}

Op *ArgMax::materialize(FFModel &ff,
                        ParallelTensor inputs[],
                        int num_inputs) const {
  ArgMaxParams params = get_params();
  return new ArgMax(ff, params, inputs[0], this->name);
}

bool ArgMax::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ArgMaxParams>::operator()(
    FlexFlow::ArgMaxParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.beam_search);
  return key;
}
}; // namespace std