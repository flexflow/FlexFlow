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

#include "flexflow/ops/gumbel_topk.h"
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
// (resp. vector along the last dimension) using Gumbel trick
// (https://arxiv.org/abs/1903.06059). Thus, values.shape = indices.shape =
// input.shape[:-1] + [k]
Tensor FFModel::gumbel_top_k(Tensor const input,
                             int k,
                             bool sorted,
                             bool speculative_decoding,
                             char const *name) {
  Layer *li = new Layer(this,
                        OP_GUMBEL_TOPK,
                        input->data_type,
                        name,
                        1,
                        0,
                        speculative_decoding ? 3 : 1 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = k;
    // token_ids
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, false /*create_grad*/);
    if (speculative_decoding) {
      // log_probs
      li->outputs[1] = create_tensor_legion_ordering(
          numdims, dims, DT_FLOAT, li, 1, false /*create_grad*/);
      // perturbed_log_probs
      li->outputs[2] = create_tensor_legion_ordering(
          numdims, dims, DT_FLOAT, li, 2, false /*create_grad*/);
    }
  }
  li->add_int_property("k", k);
  li->add_int_property("sorted", sorted);
  li->add_int_property("speculative_decoding", speculative_decoding);
  layers.push_back(li);
  return li->outputs[0];
}

Op *GumbelTopK::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("k", value);
  int k = value;
  layer->get_int_property("sorted", value);
  bool sorted = (bool)value;
  layer->get_int_property("speculative_decoding", value);
  bool speculative_decoding = (bool)value;

  return new GumbelTopK(model,
                        layer->layer_guid,
                        inputs[0],
                        k,
                        sorted,
                        speculative_decoding,
                        layer->name);
}

GumbelTopKParams GumbelTopK::get_params() const {
  GumbelTopKParams params;
  params.k = this->k;
  params.sorted = this->sorted;
  params.speculative_decoding = this->speculative_decoding;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool GumbelTopKParams::is_valid(ParallelTensorShape const &) const {
  // gumbel_topk is always valid
  return true;
}

bool operator==(GumbelTopKParams const &lhs, GumbelTopKParams const &rhs) {
  return lhs.k == rhs.k && lhs.sorted == rhs.sorted &&
         lhs.speculative_decoding == rhs.speculative_decoding;
}

GumbelTopK::GumbelTopK(FFModel &model,
                       LayerID const &_layer_guid,
                       ParallelTensor const _input,
                       int _k,
                       bool _sorted,
                       bool _speculative_decoding,
                       char const *name)
    : Op(model,
         OP_GUMBEL_TOPK,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         _speculative_decoding ? 3 : 1 /*outputs*/,
         _input),
      k(_k), sorted(_sorted), speculative_decoding(_speculative_decoding) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = inputs[0]->dims[i];
  }

  dims[0].size = k;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_INT32, this, 0 /*owner_idx*/);
  if (_speculative_decoding) {
    outputs[1] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, DT_FLOAT, this, 1 /*owner_idx*/);
    outputs[2] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, DT_FLOAT, this, 2 /*owner_idx*/);
  }
}

GumbelTopK::GumbelTopK(FFModel &model,
                       LayerID const &layer_guid,
                       GumbelTopK const &other,
                       ParallelTensor const input)
    : GumbelTopK(model,
                 layer_guid,
                 input,
                 other.k,
                 other.sorted,
                 other.speculative_decoding,
                 other.name) {}

GumbelTopK::GumbelTopK(FFModel &model,
                       GumbelTopKParams const &params,
                       ParallelTensor const input,
                       char const *name)
    : GumbelTopK(model,
                 params.layer_guid,
                 input,
                 params.k,
                 params.sorted,
                 params.speculative_decoding,
                 params.name) {}

void GumbelTopK::init_inference(
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
  IndexLauncher launcher(GUMBEL_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(GumbelTopK)),
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
  //   launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
  //                                                     0 /*projection id*/,
  //                                                     WRITE_ONLY,
  //                                                     EXCLUSIVE,
  //                                                     batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void GumbelTopK::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(GUMBEL_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(GumbelTopK)),
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *GumbelTopK::init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  GumbelTopK *gumbel_topk = (GumbelTopK *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  GumbelTopKMeta *m =
      new GumbelTopKMeta(handle, gumbel_topk, gpu_mem_allocator);
  m->profiling = gumbel_topk->profiling;
  m->inference_debugging = gumbel_topk->inference_debugging;
  m->sorted = gumbel_topk->sorted;
  m->k = gumbel_topk->k;
  std::strcpy(m->op_name, gumbel_topk->name);
  m->layer_guid = gumbel_topk->layer_guid;
  m->speculative_decoding = gumbel_topk->speculative_decoding;
  return m;
}

void GumbelTopK::forward(FFModel const &ff) {
  // GumbelTopK does not support forward
  assert(false);
}

FutureMap GumbelTopK::inference(
    FFModel const &ff,
    /* Reserved: BatchConfig Updated */ BatchConfigFuture const &bc,
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
  /* std::cout << "GumbelTopK op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  if (speculative_decoding) {
    IndexLauncher launcher(GUMBEL_TOPK_INF_SPECULATIVE_TASK_ID,
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

    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[2]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[2]->region));
    launcher.add_field(3, FID_DATA);

    return runtime->execute_index_space(ctx, launcher);
  } else {
    IndexLauncher launcher(GUMBEL_TOPK_INF_TASK_ID,
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

InferenceResult
    GumbelTopK::inference_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const GumbelTopK* topk = (const GumbelTopK*) task->args;
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }
  GumbelTopKMeta *m = *((GumbelTopKMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW log_probs;
  GenericTensorAccessorW perturbed_log_probs;

  int batch_size = bc->num_active_tokens();
  GumbelTopK::forward_kernel_wrapper(
      m, input, log_probs, perturbed_log_probs, indices, batch_size, nullptr);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    GumbelTopK::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {}, {indices});
  }

  InferenceResult ir;
  ir.num_token_ids = batch_size * m->k;
  ir.num_gumbel_logits = batch_size * m->k;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size);
  return ir;
}

InferenceResult GumbelTopK::inference_speculative_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_tokens() == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }
  GumbelTopKMeta *m = *((GumbelTopKMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW log_probs = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorW perturbed_log_probs = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[3], task->regions[3], FID_DATA, ctx, runtime);

  int batch_size = bc->num_active_tokens();
  GumbelTopK::forward_kernel_wrapper(
      m, input, log_probs, perturbed_log_probs, indices, batch_size, bc);

  InferenceResult ir;
  ir.num_token_ids = batch_size * m->k;
  ir.num_gumbel_logits = batch_size * m->k;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size * m->k);
  download_tensor<float>(
      log_probs.get_float_ptr(), ir.probs, batch_size * m->k);
  download_tensor<float>(
      perturbed_log_probs.get_float_ptr(), ir.gumbel_logits, batch_size * m->k);
  return ir;
}

void GumbelTopK::backward(FFModel const &ff) {
  // GumbelTopK does not support backward
  assert(false);
}

void GumbelTopK::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->k);
  sez.serialize(this->sorted);
  sez.serialize(this->speculative_decoding);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node GumbelTopK::deserialize(FFModel &ff,
                             Legion::Deserializer &dez,
                             ParallelTensor inputs[],
                             int num_inputs) {
  assert(num_inputs == 1);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  int k;
  bool sorted;
  bool speculative_decoding;
  dez.deserialize(k);
  dez.deserialize(sorted);
  dez.deserialize(speculative_decoding);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  GumbelTopKParams params;
  params.layer_guid = layer_guid;
  params.k = k;
  params.sorted = sorted;
  params.speculative_decoding = speculative_decoding;
  strcpy(params.name, name);
  return ff.get_or_create_node<GumbelTopK>(inputs[0], params);
}

Op *GumbelTopK::materialize(FFModel &ff,
                            ParallelTensor inputs[],
                            int num_inputs) const {
  GumbelTopKParams params = get_params();
  return new GumbelTopK(ff, params, inputs[0], this->name);
}

bool GumbelTopK::measure_operator_cost(Simulator *sim,
                                       MachineView const &mv,
                                       CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::GumbelTopKParams>::operator()(
    FlexFlow::GumbelTopKParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.k);
  hash_combine(key, params.sorted);
  hash_combine(key, params.speculative_decoding);
  return key;
}
}; // namespace std
