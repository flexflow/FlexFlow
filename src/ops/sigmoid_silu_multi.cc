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

#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

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

bool operator==(SigmoidSiluMultiParams const &lhs,
                SigmoidSiluMultiParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid;
}

bool SigmoidSiluMultiParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  return input.first.is_valid() && input.second.is_valid();
}

SigmoidSiluMultiParams SigmoidSiluMulti::get_params() const {
  SigmoidSiluMultiParams params;
  params.layer_guid = this->layer_guid;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

Tensor FFModel::sigmoid_silu_multi(const Tensor input1,
                                   const Tensor input2,
                                   DataType data_type,
                                   char const *name) {

  // Check dims
  assert(input1->num_dims == input2->num_dims);
  for (int i = 0; i < input1->num_dims; i++) {
    assert(input1->dims[i] == input2->dims[i]);
  }
  // Tensor Data type
  if (data_type == DT_NONE) {
    data_type = input1->data_type;
    assert(input2->data_type == input1->data_type);
  }
  Tensor casted_input1 =
      (data_type != input1->data_type)
          ? cast(input1, data_type, "type cast for sigmoid_silu_multi")
          : input1;
  Tensor casted_input2 =
      (data_type != input2->data_type)
          ? cast(input2, data_type, "type cast for sigmoid_silu_multi")
          : input2;

  // Create layer
  Layer *ssm = new Layer(this,
                         OP_SIGMOID_SILU_MULTI,
                         data_type,
                         name,
                         2 /*inputs*/,
                         0 /*weights*/,
                         1 /*outputs*/,
                         casted_input1,
                         casted_input2);
  ssm->outputs[0] = create_tensor_legion_ordering(
      input1->num_dims, input1->dims, data_type, ssm, 0, false /*create_grad*/);
  layers.push_back(ssm);
  return ssm->outputs[0];
}

Op *SigmoidSiluMulti::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {

  return new SigmoidSiluMulti(
      model, layer->layer_guid, inputs[0], inputs[1], layer->name);
}

SigmoidSiluMulti::SigmoidSiluMulti(
    FFModel &model,
    SigmoidSiluMultiParams const &params,
    std::pair<ParallelTensor, ParallelTensor> const &inputs,
    char const *name)
    : SigmoidSiluMulti(
          model, params.layer_guid, inputs.first, inputs.second, params.name) {}

SigmoidSiluMulti::SigmoidSiluMulti(FFModel &model,
                                   LayerID const &_layer_guid,
                                   const ParallelTensor _input1,
                                   const ParallelTensor _input2,
                                   char const *name)
    : Op(model,
         OP_SIGMOID_SILU_MULTI,
         _input1->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input1,
         _input2) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(_input1->num_dims,
                                                            _input1->dims,
                                                            _input1->data_type,
                                                            this,
                                                            0 /*owner_idx*/);
}

void SigmoidSiluMulti::init_inference(
    FFModel const &ff,
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
  IndexLauncher launcher(SIGMOID_SILU_MULTI_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(SigmoidSiluMulti)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // input 1
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // input 2
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // output
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void SigmoidSiluMulti::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(SIGMOID_SILU_MULTI_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(SigmoidSiluMulti)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // input 1
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // input 2
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0](I): input 1
  regions[1](I): input 2
  regions[2](O): output
*/
OpMeta *SigmoidSiluMulti::init_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  SigmoidSiluMulti *ssm = (SigmoidSiluMulti *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  SigmoidSiluMultiMeta *meta =
      new SigmoidSiluMultiMeta(handle, ssm, gpu_mem_allocator);
  meta->input_type[0] = ssm->inputs[0]->data_type;
  meta->input_type[1] = ssm->inputs[1]->data_type;
  meta->output_type[0] = ssm->outputs[0]->data_type;
  std::strcpy(meta->op_name, ssm->name);
  meta->layer_guid = ssm->layer_guid;
  return meta;
}

void SigmoidSiluMulti::forward(FFModel const &ff) {
  assert(false);
}

void SigmoidSiluMulti::backward(FFModel const &ff) {
  assert(false);
}

FutureMap SigmoidSiluMulti::inference(
    FFModel const &ff,
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
  /* std::cout << "SigmoidSiluMulti op machine_view: " << *(MachineView
     const *)mv
            << std::endl; */
  IndexLauncher launcher(SIGMOID_SILU_MULTI_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  // input 1
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // input 2
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // output
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input 1
  regions[1](I): input 2
  regions[2](O): output
*/
void SigmoidSiluMulti::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {

  assert(task->regions.size() == regions.size());
  assert(regions.size() == 3);

  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }

  SigmoidSiluMultiMeta *m = *((SigmoidSiluMultiMeta **)task->local_args);

  GenericTensorAccessorR input1 = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR input2 = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);

  Domain input1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain input2_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  assert(input1_domain.get_volume() == input2_domain.get_volume());
  assert(input1_domain.get_volume() == output_domain.get_volume());

  assert(input1_domain == input2_domain);
  assert(input1_domain == output_domain);

  SigmoidSiluMulti::inference_kernel_wrapper(m, input1, input2, output);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    SigmoidSiluMulti::save_inference_tensors_to_file(
        m, shard_id, bc, {input1, input2}, {}, {output});
  }
}

bool SigmoidSiluMulti::measure_operator_cost(Simulator *sim,
                                             MachineView const &mv,
                                             CostMetrics &cost_metrics) const {
  return false;
}

void SigmoidSiluMulti::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node SigmoidSiluMulti::deserialize(FFModel &ff,
                                   Legion::Deserializer &dez,
                                   ParallelTensor inputs[],
                                   int num_inputs) {
  assert(num_inputs == 2);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  SigmoidSiluMultiParams params;
  params.layer_guid = layer_guid;
  strcpy(params.name, name);
  return ff.get_or_create_node<SigmoidSiluMulti>({inputs[0], inputs[1]},
                                                 params);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SigmoidSiluMultiParams>::operator()(
    FlexFlow::SigmoidSiluMultiParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.layer_guid.transformer_layer_id);
  hash_combine(key, params.layer_guid.model_id);
  return key;
}
}; // namespace std
