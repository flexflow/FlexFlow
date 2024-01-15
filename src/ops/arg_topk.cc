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

#include "flexflow/ops/arg_topk.h"
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
Tensor FFModel::arg_top_k(const Tensor input,
                          int k,
                          bool sorted,
                          bool speculative_decoding,
                          char const *name) {
  Layer *li = new Layer(this,
                        OP_ARG_TOPK,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        speculative_decoding ? 2 : 1 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = k;
    // li->outputs[0] = create_tensor_legion_ordering(
    //     numdims, dims, input->data_type, li, 0, true /*create_grad*/);
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, false /*create_grad*/);
    if (speculative_decoding) {
      li->outputs[1] = create_tensor_legion_ordering(
          numdims, dims, DT_FLOAT, li, 1, false /*create_grad*/);
    }
  }
  li->add_int_property("k", k);
  li->add_int_property("sorted", sorted);
  li->add_int_property("speculative_decoding", speculative_decoding);
  layers.push_back(li);
  // outputs[0] = li->outputs[0];
  // outputs[1] = li->outputs[1];
  return li->outputs[0];
}

Op *ArgTopK::create_operator_from_layer(
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

  return new ArgTopK(model,
                     layer->layer_guid,
                     inputs[0],
                     k,
                     sorted,
                     speculative_decoding,
                     layer->name);
}

ArgTopKParams ArgTopK::get_params() const {
  ArgTopKParams params;
  params.k = this->k;
  params.sorted = this->sorted;
  params.speculative_decoding = this->speculative_decoding;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool ArgTopKParams::is_valid(ParallelTensorShape const &) const {
  // topk is always valid
  return true;
}

bool operator==(ArgTopKParams const &lhs, ArgTopKParams const &rhs) {
  return lhs.k == rhs.k && lhs.sorted == rhs.sorted &&
         lhs.speculative_decoding == rhs.speculative_decoding;
}

ArgTopK::ArgTopK(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor _input,
                 int _k,
                 bool _sorted,
                 bool _speculative_decoding,
                 char const *name)
    : Op(model,
         OP_ARG_TOPK,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         _speculative_decoding ? 2 : 1 /*outputs*/,
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
  }
}

ArgTopK::ArgTopK(FFModel &model,
                 LayerID const &layer_guid,
                 ArgTopK const &other,
                 const ParallelTensor input)
    : ArgTopK(model,
              layer_guid,
              input,
              other.k,
              other.sorted,
              other.speculative_decoding,
              other.name) {}

ArgTopK::ArgTopK(FFModel &model,
                 ArgTopKParams const &params,
                 ParallelTensor const input,
                 char const *name)
    : ArgTopK(model,
              params.layer_guid,
              input,
              params.k,
              params.sorted,
              params.speculative_decoding,
              params.name) {}

void ArgTopK::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(ARG_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ArgTopK)),
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
  //   launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void ArgTopK::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ARG_TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ArgTopK)),
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
  //   launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
  //                                                     0 /*projection id*/,
  //                                                     WRITE_ONLY,
  //                                                     EXCLUSIVE,
  //                                                     outputs[1]->region));
  //   launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *ArgTopK::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  ArgTopK *topk = (ArgTopK *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  ArgTopKMeta *m = new ArgTopKMeta(handle, topk);
  m->profiling = topk->profiling;
  m->inference_debugging = topk->inference_debugging;
  m->sorted = topk->sorted;
  m->k = topk->k;
  std::strcpy(m->op_name, topk->name);
  m->layer_guid = topk->layer_guid;
  m->speculative_decoding = topk->speculative_decoding;
  return m;
}

void ArgTopK::forward(FFModel const &ff) {
  // ArgTopK does not support forward
  assert(false);
}

FutureMap ArgTopK::inference(FFModel const &ff,
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
  /* std::cout << "ArgTopK op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  if (speculative_decoding) {
    IndexLauncher launcher(ARG_TOPK_INF_SPECULATIVE_TASK_ID,
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
    return runtime->execute_index_space(ctx, launcher);

  } else {
    IndexLauncher launcher(ARG_TOPK_INF_TASK_ID,
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
    ArgTopK::inference_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const ArgTopK* topk = (const ArgTopK*) task->args;
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    // Directly return for empty batch config
    InferenceResult ir;
    return ir;
  }
  ArgTopKMeta *m = *((ArgTopKMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW probs;

  int batch_size = bc->num_active_tokens();
  ArgTopK::forward_kernel_wrapper(
      m, input, probs, indices, batch_size, nullptr);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    ArgTopK::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {}, {indices});
  }

  InferenceResult ir;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size);
  return ir;
}

BeamInferenceResult ArgTopK::inference_speculative_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  BeamSearchBatchConfig const &bc =
      Future(task->futures[0]).get_result<BeamSearchBatchConfig>();
  if (bc.num_active_tokens() == 0) {
    // Directly return for empty batch config
    BeamInferenceResult ir;
    return ir;
  }
  ArgTopKMeta *m = *((ArgTopKMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW indices = helperGetGenericTensorAccessorWO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW probs = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int batch_size = bc.num_active_tokens();
  ArgTopK::forward_kernel_wrapper(m, input, probs, indices, batch_size, &bc);

  BeamInferenceResult ir;
  download_tensor<BatchConfig::TokenId>(
      indices.get_int32_ptr(), ir.token_ids, batch_size * m->k);
  download_tensor<float>(probs.get_float_ptr(), ir.probs, batch_size * m->k);
  return ir;
}

void ArgTopK::backward(FFModel const &ff) {
  // ArgTopK does not support backward
  assert(false);
}

void ArgTopK::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->k);
  sez.serialize(this->sorted);
  sez.serialize(this->speculative_decoding);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node ArgTopK::deserialize(FFModel &ff,
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
  ArgTopKParams params;
  params.layer_guid = layer_guid;
  params.k = k;
  params.sorted = sorted;
  params.speculative_decoding = speculative_decoding;
  strcpy(params.name, name);
  return ff.get_or_create_node<ArgTopK>(inputs[0], params);
}

Op *ArgTopK::materialize(FFModel &ff,
                         ParallelTensor inputs[],
                         int num_inputs) const {
  ArgTopKParams params = get_params();
  return new ArgTopK(ff, params, inputs[0], this->name);
}

bool ArgTopK::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ArgTopKParams>::operator()(
    FlexFlow::ArgTopKParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.k);
  hash_combine(key, params.sorted);
  hash_combine(key, params.speculative_decoding);
  return key;
}
}; // namespace std
