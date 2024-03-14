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

#include "flexflow/ops/residual_layer_norm.h"
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

bool operator==(ResidualLayerNormParams const &lhs,
                ResidualLayerNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.axes == rhs.axes &&
         lhs.elementwise_affine == rhs.elementwise_affine &&
         lhs.use_bias == rhs.use_bias &&
         lhs.use_two_residuals == rhs.use_two_residuals;
}

bool ResidualLayerNormParams::is_valid(
    std::tuple<ParallelTensorShape,
               ParallelTensorShape,
               ParallelTensorShape> const &input) const {
  return std::get<0>(input).is_valid() && std::get<1>(input).is_valid() &&
         (!use_two_residuals || std::get<2>(input).is_valid());
}

ResidualLayerNormParams ResidualLayerNorm::get_params() const {
  ResidualLayerNormParams params;
  params.layer_guid = this->layer_guid;
  params.axes = this->axes;
  params.elementwise_affine = this->elementwise_affine;
  params.eps = this->eps;
  params.use_bias = this->use_bias;
  params.use_two_residuals = this->use_two_residuals;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

void FFModel::residual_layer_norm(const Tensor input,
                                  const Tensor residual1,
                                  const Tensor residual2,
                                  Tensor *outputs,
                                  bool use_two_residuals,
                                  std::vector<int> const &axes,
                                  bool elementwise_affine,
                                  float eps,
                                  bool use_bias,
                                  DataType data_type,
                                  char const *name) {
  // In PyTorch, axes must be the sizes of the last axes.size() dimensions of
  // the input tensor. However, since the tensor dimensions are reversed in
  // FlexFlow (batch size is the last dimension), we require that axes must be
  // the sizes of the FIRST axes.size() dimensions of the input tensor.

  // Another difference is that in PyTorch, the axes vector should contain the
  // sizes of the dimensions with respect to which you want to compute the
  // layernorm. In FlexFlow, instead, axes should contain the INDICES of the
  // dimensions in question. We do this because the size of a dimension might be
  // different when splitting a tensor in model parallelism.
  assert(
      axes.size() <= input->num_dims &&
      "number of axes must be less than tensor dimensions"); // input does not
                                                             // have replica
                                                             // dimension here
  for (int i = 0; i < axes.size(); i++) {
    assert(axes[i] == i && "axes must be the first axes.size() dimensions");
  }

  // Check dims
  assert(input->num_dims == residual1->num_dims);
  if (use_two_residuals) {
    assert(residual2 != nullptr);
    assert(input->num_dims == residual2->num_dims);
  }
  for (int i = 0; i < input->num_dims; i++) {
    assert(input->dims[i] == residual1->dims[i]);
    if (use_two_residuals) {
      assert(input->dims[i] == residual2->dims[i]);
    }
  }

  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }

  int num_weights = elementwise_affine ? (use_bias ? 2 : 1) : 0;
  Layer *ln = nullptr;
  Tensor casted_input =
      (data_type != input->data_type)
          ? cast(input, data_type, "type cast for residual_layer_norm")
          : input;
  Tensor casted_residual1 =
      (data_type != residual1->data_type)
          ? cast(residual1, data_type, "type cast for residual1_layer_norm")
          : residual1;
  Tensor casted_residual2 = nullptr;
  if (use_two_residuals) {
    casted_residual2 =
        (data_type != residual2->data_type)
            ? cast(residual2, data_type, "type cast for residual2_layer_norm")
            : residual2;
  }
  ln = new Layer(this,
                 OP_RESIDUAL_LAYERNORM,
                 data_type,
                 name,
                 2 + use_two_residuals /*inputs*/,
                 num_weights,
                 2 /*outputs*/,
                 casted_input,
                 casted_residual1,
                 casted_residual2);
  ln->outputs[0] = create_tensor_legion_ordering(
      input->num_dims, input->dims, data_type, ln, 0, false /*create_grad*/);
  ln->outputs[1] = create_tensor_legion_ordering(
      input->num_dims, input->dims, data_type, ln, 1, false /*create_grad*/);
  {
    int numdims = axes.size();
    int dims[numdims];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[axes[i]];
    }
    if (num_weights >= 1) {
      assert(elementwise_affine);
      ln->weights[0] = create_weight_legion_ordering(numdims,
                                                     dims,
                                                     data_type,
                                                     ln,
                                                     false /*create_grad*/,
                                                     nullptr,
                                                     CHOSEN_SYNC_TYPE);
      if (num_weights == 2) {
        assert(use_bias);
        ln->weights[1] = create_weight_legion_ordering(numdims,
                                                       dims,
                                                       data_type,
                                                       ln,
                                                       false /*create_grad*/,
                                                       nullptr,
                                                       CHOSEN_SYNC_TYPE);
      }
    }
  }
  ln->add_int_property("elementwise_affine", elementwise_affine);
  ln->add_int_property("use_bias", use_bias);
  ln->add_int_vector_property("axes", axes);
  ln->add_float_property("eps", eps);
  ln->add_int_property("use_two_residuals", use_two_residuals);
  layers.push_back(ln);
  outputs[0] = ln->outputs[0];
  outputs[1] = ln->outputs[1];
}

Op *ResidualLayerNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("elementwise_affine", value);
  bool elementwise_affine = (bool)value;
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  std::vector<int> axes;
  layer->get_int_vector_property("axes", axes);
  float eps;
  layer->get_float_property("eps", eps);
  layer->get_int_property("use_two_residuals", value);
  bool use_two_residuals = (bool)value;
  return new ResidualLayerNorm(model,
                               layer->layer_guid,
                               inputs[0],
                               inputs[1],
                               use_two_residuals ? inputs[2] : nullptr,
                               use_two_residuals,
                               axes,
                               elementwise_affine,
                               use_bias,
                               eps,
                               false, // allocate_weights
                               layer->name);
}

ResidualLayerNorm::ResidualLayerNorm(
    FFModel &model,
    ResidualLayerNormParams const &params,
    std::tuple<ParallelTensor, ParallelTensor, ParallelTensor> const &inputs,
    bool allocate_weights,
    char const *name)
    : ResidualLayerNorm(model,
                        params.layer_guid,
                        std::get<0>(inputs),
                        std::get<1>(inputs),
                        params.use_two_residuals ? std::get<2>(inputs)
                                                 : nullptr,
                        params.use_two_residuals,
                        params.axes,
                        params.elementwise_affine,
                        params.use_bias,
                        params.eps,
                        allocate_weights,
                        params.name) {}

ResidualLayerNorm::ResidualLayerNorm(FFModel &model,
                                     LayerID const &_layer_guid,
                                     const ParallelTensor _input,
                                     const ParallelTensor _residual1,
                                     const ParallelTensor _residual2,
                                     bool _use_two_residuals,
                                     std::vector<int> const &_axes,
                                     bool _elementwise_affine,
                                     bool _use_bias,
                                     float _eps,
                                     bool allocate_weights,
                                     char const *name)
    : Op(model,
         OP_RESIDUAL_LAYERNORM,
         _input->data_type,
         name,
         2 + _use_two_residuals /*inputs*/,
         _elementwise_affine ? (_use_bias ? 2 : 1) : 0 /*weights*/,
         2 /*outputs*/,
         _input,
         _residual1,
         _use_two_residuals ? _residual2 : nullptr),
      elementwise_affine(_elementwise_affine), eps(_eps), axes(_axes),
      use_bias(_use_bias), use_two_residuals(_use_two_residuals) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this, 0 /*owner_idx*/);
  outputs[1] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this, 1 /*owner_idx*/);
  assert(check_output_input_weight_parallel_dims(allocate_weights));

  int M = 1;
  for (int i = 0; i < axes.size(); i++) {
    M *= inputs[0]->dims[axes[i]].size;
  }
  int num_replicas = 1;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    if (inputs[0]->dims[i].is_replica_dim) {
      num_replicas *= inputs[0]->dims[i].size;
    }
  }
  effective_num_elements = M;
  effective_batch_size = (inputs[0]->get_volume() / num_replicas) / M;
  if (!elementwise_affine) {
    assert(numWeights == 0);
  } else {
    if (!use_bias) {
      assert(numWeights == 1); // weight
    } else {
      assert(numWeights == 2); // weight + bias
    }
  }

  if (allocate_weights) {
    int seed = std::rand();
    if (numWeights >= 1) {
      assert(elementwise_affine);

      ParallelTensorShape beta_gamma_shape = _input->get_shape();
      for (int i = axes.size(); i < beta_gamma_shape.num_dims - 1; i++) {
        beta_gamma_shape.dims[i].size = 1;
      }

      // weight
      Initializer *gamma_initializer = new UniformInitializer(seed, 1.0f, 1.0f);
      weights[0] = model.create_parallel_weight_legion_ordering(
          beta_gamma_shape.num_dims, // axes.size(),
          beta_gamma_shape.dims,
          _input->data_type,
          NULL /*owner_op*/,
          false /*create_grad*/,
          gamma_initializer,
          CHOSEN_SYNC_TYPE);

      // bias
      if (numWeights == 2) {
        assert(use_bias);
        Initializer *beta_initializer =
            new UniformInitializer(seed, 0.0f, 0.0f);
        weights[1] = model.create_parallel_weight_legion_ordering(
            beta_gamma_shape.num_dims, //.size(),
            beta_gamma_shape.dims,
            _input->data_type,
            NULL /*owner_op*/,
            false /*create_grad*/,
            beta_initializer,
            CHOSEN_SYNC_TYPE);
      }
    }
  }
}

void ResidualLayerNorm::init_inference(
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
  IndexLauncher launcher(RESIDUAL_LAYERNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ResidualLayerNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  int field_id = 0;
  // input
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual1
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual2
  if (use_two_residuals) {
    launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[2]->region));
    launcher.add_field(field_id++, FID_DATA);
  }
  // added: input + residual(s)
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // layer norm output
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  // weights
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(field_id++, FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(field_id++, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void ResidualLayerNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ADD_BIAS_RESIDUAL_LAYERNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ResidualLayerNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  int field_id = 0;
  // input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual1
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual2
  if (use_two_residuals) {
    launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[1]->region));
    launcher.add_field(field_id++, FID_DATA);
  }
  // added: input + residual(s)
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // layer norm output
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  // weights
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(field_id++, FID_DATA);

    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(field_id++, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *ResidualLayerNorm::init_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  ResidualLayerNorm *ln = (ResidualLayerNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  ResidualLayerNormMeta *meta =
      new ResidualLayerNormMeta(handle, ln, gpu_mem_allocator);
  std::strcpy(meta->op_name, ln->name);
  meta->layer_guid = ln->layer_guid;
  meta->input_type[0] = ln->inputs[0]->data_type;
  meta->input_type[1] = ln->inputs[1]->data_type;
  if (ln->use_two_residuals) {
    meta->input_type[2] = ln->inputs[2]->data_type;
  }
  if (ln->elementwise_affine) {
    meta->weight_type[0] = ln->weights[0]->data_type;
    if (ln->use_bias) {
      meta->weight_type[1] = ln->weights[1]->data_type;
    }
  }
  meta->output_type[0] = ln->outputs[0]->data_type;
  meta->output_type[1] = ln->outputs[1]->data_type;
  return meta;
}

void ResidualLayerNorm::forward(FFModel const &ff) {
  assert(false);
}

void ResidualLayerNorm::backward(FFModel const &ff) {
  assert(false);
}

Op *ResidualLayerNorm::materialize(FFModel &ff,
                                   ParallelTensor inputs[],
                                   int num_inputs) const {
  ResidualLayerNormParams params = get_params();
  return new ResidualLayerNorm(
      ff,
      params,
      {inputs[0], inputs[1], params.use_two_residuals ? inputs[2] : nullptr},
      true,
      this->name);
}

FutureMap ResidualLayerNorm::inference(
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

  IndexLauncher launcher(RESIDUAL_LAYERNORM_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  int field_id = 0;
  // input
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual1
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  // residual2
  if (use_two_residuals) {
    launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[2]->region));
    launcher.add_field(field_id++, FID_DATA);
  }
  // added: input + residual(s)
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(field_id++, FID_DATA);
  // layer norm output
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(field_id++, FID_DATA);
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(field_id++, FID_DATA);

    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(field_id++, FID_DATA);
    }
  }
  return runtime->execute_index_space(ctx, launcher);
}

void ResidualLayerNorm::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {

  assert(task->regions.size() == regions.size());
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }

  ResidualLayerNormMeta *m = *((ResidualLayerNormMeta **)task->local_args);

  assert(regions.size() ==
         4 + m->use_two_residuals +
             (m->elementwise_affine ? (m->use_bias ? 2 : 1) : 0));

  int region_idx = 0, task_region_idx = 0;
  GenericTensorAccessorR input =
      helperGetGenericTensorAccessorRO(m->input_type[0],
                                       regions[region_idx++],
                                       task->regions[task_region_idx++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR residual1 =
      helperGetGenericTensorAccessorRO(m->input_type[1],
                                       regions[region_idx++],
                                       task->regions[task_region_idx++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR residual2;
  if (m->use_two_residuals) {
    residual2 =
        helperGetGenericTensorAccessorRO(m->input_type[2],
                                         regions[region_idx++],
                                         task->regions[task_region_idx++],
                                         FID_DATA,
                                         ctx,
                                         runtime);
  }
  GenericTensorAccessorW added_output =
      helperGetGenericTensorAccessorWO(m->output_type[0],
                                       regions[region_idx++],
                                       task->regions[task_region_idx++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorWO(m->output_type[1],
                                       regions[region_idx++],
                                       task->regions[task_region_idx++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR gamma, beta;
  if (m->elementwise_affine) {
    gamma = helperGetGenericTensorAccessorRO(m->weight_type[0],
                                             regions[region_idx++],
                                             task->regions[task_region_idx++],
                                             FID_DATA,
                                             ctx,
                                             runtime);
    if (m->use_bias) {
      beta = helperGetGenericTensorAccessorRO(m->weight_type[1],
                                              regions[region_idx++],
                                              task->regions[task_region_idx++],
                                              FID_DATA,
                                              ctx,
                                              runtime);
    }
  }

  task_region_idx = 0;
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[task_region_idx++].region.get_index_space());
  Domain residual1_domain = runtime->get_index_space_domain(
      ctx, task->regions[task_region_idx++].region.get_index_space());
  Domain residual2_domain;
  if (m->use_two_residuals) {
    residual2_domain = runtime->get_index_space_domain(
        ctx, task->regions[task_region_idx++].region.get_index_space());
    assert(in_domain.get_volume() == residual2_domain.get_volume());
    assert(residual2_domain == in_domain);
  }
  Domain added_out_domain = runtime->get_index_space_domain(
      ctx, task->regions[task_region_idx++].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[task_region_idx++].region.get_index_space());
  Domain gamma_domain, beta_domain;
  if (m->elementwise_affine) {
    gamma_domain = runtime->get_index_space_domain(
        ctx, task->regions[task_region_idx++].region.get_index_space());
    assert(gamma_domain.get_volume() == m->effective_num_elements);
    int numdims = gamma_domain.get_dim();
    size_t vol = 1;
    int i = 0;
    while (vol < gamma_domain.get_volume()) {
      int g_d = gamma_domain.hi()[i] - gamma_domain.lo()[i] + 1;
      int in_d = in_domain.hi()[i] - in_domain.lo()[i] + 1;
      assert(g_d == in_d);
      vol *= g_d;
      i++;
    }
    if (m->use_bias) {
      beta_domain = runtime->get_index_space_domain(
          ctx, task->regions[task_region_idx++].region.get_index_space());
      assert(gamma_domain == beta_domain);
    }
  }
  assert(in_domain.get_volume() == out_domain.get_volume());
  assert(out_domain.get_volume() == added_out_domain.get_volume());
  assert(in_domain.get_volume() == residual1_domain.get_volume());
  assert(in_domain == out_domain);
  assert(added_out_domain == out_domain);
  assert(residual1_domain == in_domain);
  assert(in_domain.get_volume() ==
         m->effective_num_elements * m->effective_batch_size);

  ResidualLayerNorm::inference_kernel_wrapper(
      m, input, residual1, residual2, added_output, output, gamma, beta);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    std::vector<GenericTensorAccessorR> input_accessors;
    input_accessors.push_back(input);
    input_accessors.push_back(residual1);
    if (m->use_two_residuals) {
      input_accessors.push_back(residual2);
    }
    std::vector<GenericTensorAccessorR> weights_accessors;
    if (m->elementwise_affine) {
      weights_accessors.push_back(gamma);
      if (m->use_bias) {
        weights_accessors.push_back(beta);
      }
    }
    ResidualLayerNorm::save_inference_tensors_to_file(m,
                                                      shard_id,
                                                      bc,
                                                      input_accessors,
                                                      weights_accessors,
                                                      {added_output, output});
  }
}

bool ResidualLayerNorm::measure_operator_cost(Simulator *sim,
                                              MachineView const &mv,
                                              CostMetrics &cost_metrics) const {
  return false;
}

void ResidualLayerNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->axes.size());
  for (size_t i = 0; i < this->axes.size(); i++) {
    sez.serialize(this->axes[i]);
  }
  sez.serialize(this->elementwise_affine);
  sez.serialize(this->eps);
  sez.serialize(this->use_bias);
  sez.serialize(this->use_two_residuals);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node ResidualLayerNorm::deserialize(FFModel &ff,
                                    Legion::Deserializer &dez,
                                    ParallelTensor inputs[],
                                    int num_inputs) {
  size_t num_axes;
  std::vector<int> axes;
  bool elementwise_affine;
  bool use_bias;
  bool use_two_residuals;
  float eps;
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(num_axes);
  for (size_t i = 0; i < num_axes; i++) {
    int axis_idx;
    dez.deserialize(axis_idx);
    axes.push_back(axis_idx);
  }
  dez.deserialize(elementwise_affine);
  dez.deserialize(eps);
  dez.deserialize(use_bias);
  dez.deserialize(use_two_residuals);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  if (use_two_residuals) {
    assert(num_inputs == 3);
  } else {
    assert(num_inputs == 2);
  }

  ResidualLayerNormParams params;
  params.layer_guid = layer_guid;
  params.axes = axes;
  params.elementwise_affine = elementwise_affine;
  params.eps = eps;
  params.use_bias = use_bias;
  params.use_two_residuals = use_two_residuals;
  strcpy(params.name, name);
  if (use_two_residuals) {
    return ff.get_or_create_node<ResidualLayerNorm>(
        {inputs[0], inputs[1], inputs[2]}, params);
  } else {
    return ff.get_or_create_node<ResidualLayerNorm>(
        {inputs[0], inputs[1], inputs[1]}, params);
  }
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ResidualLayerNormParams>::operator()(
    FlexFlow::ResidualLayerNormParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.layer_guid.transformer_layer_id);
  hash_combine(key, params.layer_guid.model_id);
  hash_combine(key, params.axes.size());
  for (int n : params.axes) {
    hash_combine(key, n);
  }
  hash_combine(key, params.elementwise_affine);
  hash_combine(key, params.use_bias);
  hash_combine(key, params.use_two_residuals);
  return key;
}
}; // namespace std
