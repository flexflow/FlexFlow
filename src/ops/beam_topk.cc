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

#include "flexflow/ops/beam_topk.h"
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
using Legion::Future;
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
Tensor FFModel::beam_top_k(const Tensor input,
                           int max_beam_width,
                           bool sorted,
                           char const *name) {
  Layer *li = new Layer(this,
                        OP_BEAM_TOPK,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        3 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;

    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = max_beam_width;

    std::cout << "beam input dimen:" << numdims << "\n";
    for (int i = 0; i < numdims; i++) {
      std::cout << input->dims[i] << ", ";
    }

    // beam width is dynamic
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, false /*create_grad*/);
    li->outputs[1] = create_tensor_legion_ordering(
        numdims, dims, DT_FLOAT, li, 1, false /*create_grad*/);
    li->outputs[2] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 1, false /*create_grad*/);
  }
  li->add_int_property("sorted", sorted);
  li->add_int_property("max_beam_width", max_beam_width);
  layers.push_back(li);
  // outputs[0] = li->outputs[0];
  // outputs[1] = li->outputs[1];
  return li->outputs[1];
}

Op *BeamTopK::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("sorted", value);
  bool sorted = (bool)value;
  layer->get_int_property("max_beam_width", value);
  int max_beam_width = value;
  return new BeamTopK(
      model, inputs[0], layer->layer_guid, max_beam_width, sorted, layer->name);
}

BeamTopKParams BeamTopK::get_params() const {
  BeamTopKParams params;
  params.layer_guid = this->layer_guid;
  params.sorted = this->sorted;
  params.max_beam_width = this->max_beam_width;
  return params;
}

bool BeamTopKParams::is_valid(ParallelTensorShape const &) const {
  // topk is always valid
  return true;
}

bool operator==(BeamTopKParams const &lhs, BeamTopKParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.sorted == rhs.sorted &&
         lhs.max_beam_width == rhs.max_beam_width;
}

BeamTopK::BeamTopK(FFModel &model,
                   const ParallelTensor _input,
                   LayerID const &_layer_guid,
                   int _max_beam_width,
                   bool _sorted,
                   char const *name)
    : Op(model,
         OP_BEAM_TOPK,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         3 /*outputs*/,
         _input) {
  sorted = _sorted;
  max_beam_width = _max_beam_width;
  layer_guid = _layer_guid;
  int numdim = inputs[0]->num_dims;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  //   outputs[0] = model.create_parallel_tensor_legion_ordering(
  //       numdim, dims, _input->data_type, this, 0 /*owner_idx*/);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, inputs[0]->dims, DT_INT32, this, 0 /*owner_idx*/);
  outputs[1] = model.create_parallel_tensor_legion_ordering(
      numdim, inputs[0]->dims, DT_FLOAT, this, 1 /*owner_idx*/);
  outputs[2] = model.create_parallel_tensor_legion_ordering(
      numdim, inputs[0]->dims, DT_INT32, this, 2 /*owner_idx*/);
}

BeamTopK::BeamTopK(FFModel &model,
                   BeamTopK const &other,
                   const ParallelTensor input)
    : BeamTopK(model,
               input,
               other.layer_guid,
               other.max_beam_width,
               other.sorted,
               other.name) {}

BeamTopK::BeamTopK(FFModel &model,
                   BeamTopKParams const &params,
                   const ParallelTensor input,
                   char const *name)
    : BeamTopK(model,
               input,
               params.layer_guid,
               params.max_beam_width,
               params.sorted,
               params.name) {}

void BeamTopK::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(BEAM_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(BeamTopK)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[2]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[2]->region));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void BeamTopK::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(BEAM_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(BeamTopK)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[2]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[2]->region));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *BeamTopK::init_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  BeamTopK *topk = (BeamTopK *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  BeamTopKMeta *m = new BeamTopKMeta(handle, topk, gpu_mem_allocator);
  m->profiling = topk->profiling;
  m->inference_debugging = topk->inference_debugging;
  std::strcpy(m->op_name, topk->name);
  m->layer_guid = topk->layer_guid;
  m->sorted = topk->sorted;
  m->max_beam_width = topk->max_beam_width;
  m->input_type[0] = topk->inputs[0]->data_type;
  return m;
}

void BeamTopK::forward(FFModel const &ff) {
  assert(false);
}

FutureMap BeamTopK::inference(FFModel const &ff,
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

  IndexLauncher launcher(BEAM_TOPK_INF_TASK_ID,
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
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[2]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[2]->region));
  launcher.add_field(3, FID_DATA);

  return runtime->execute_index_space(ctx, launcher);
}

BeamInferenceResult
    BeamTopK::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {

  assert(regions.size() == 4);
  assert(task->regions.size() == 4);

  BeamTopKMeta *m = *((BeamTopKMeta **)task->local_args);
  BeamSearchBatchConfig const &bc =
      Future(task->futures[0]).get_result<BeamSearchBatchConfig>();

  if (bc.num_tokens == 0) {
    BeamInferenceResult ir;
    return ir;
  }

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW index = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW value = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorW parent = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[3], task->regions[3], FID_DATA, ctx, runtime);

  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  int *index_ptr = index.get_int32_ptr();
  float *value_ptr = value.get_float_ptr();
  int *parent_ptr = parent.get_int32_ptr();

  // embedding size: eg. 4096
  int length = input_domain.hi()[0] - input_domain.lo()[0] + 1;
  // total token nums
  size_t batch_size = bc.num_active_tokens();

  // need meta for: how many sub requests in a main request
  BeamTopK::forward_kernel_wrapper(m,
                                   &bc,
                                   input,
                                   value_ptr,
                                   index_ptr,
                                   parent_ptr,
                                   batch_size,
                                   length,
                                   m->sorted);

  BeamInferenceResult ir;

  download_tensor<int>(index_ptr, ir.token_ids, batch_size * m->max_beam_width);
  download_tensor<float>(value_ptr, ir.probs, batch_size * m->max_beam_width);
  download_tensor<int>(
      parent_ptr, ir.parent_id, batch_size * m->max_beam_width);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    BeamTopK::save_inference_tensors_to_file(
        m, shard_id, &bc, {input}, {}, {index, value, parent});
  }

  return ir;
}

void BeamTopK::backward(FFModel const &ff) {
  assert(false);
}

void BeamTopK::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->sorted);
  sez.serialize(this->max_beam_width);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node BeamTopK::deserialize(FFModel &ff,
                           Legion::Deserializer &dez,
                           ParallelTensor inputs[],
                           int num_inputs) {
  assert(num_inputs == 1);
  bool sorted;
  size_t id, transformer_layer_id, deserialized_model_id;
  int max_beam_width;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(sorted);
  dez.deserialize(max_beam_width);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  BeamTopKParams params;
  params.layer_guid = layer_guid;
  params.sorted = sorted;
  params.max_beam_width = max_beam_width;
  strcpy(params.name, name);
  return ff.get_or_create_node<BeamTopK>(inputs[0], params);
}

Op *BeamTopK::materialize(FFModel &ff,
                          ParallelTensor inputs[],
                          int num_inputs) const {
  BeamTopKParams params = get_params();
  return new BeamTopK(ff, params, inputs[0], this->name);
}

bool BeamTopK::measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::BeamTopKParams>::operator()(
    FlexFlow::BeamTopKParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.sorted);
  hash_combine(key, params.max_beam_width);
  return key;
}
}; // namespace std
