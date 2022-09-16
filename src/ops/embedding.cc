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

#include "flexflow/ops/embedding.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::embedding(const Tensor input,
                          int num_entries,
                          int out_dim,
                          AggrMode aggr,
                          Layer const *shared_op,
                          Initializer *kernel_initializer,
                          char const *name) {
  Layer *embed = new Layer(this,
                           OP_EMBEDDING,
                           name,
                           1 /*inputs*/,
                           1 /*weights*/,
                           1 /*outputs*/,
                           input);
  if (aggr == AGGR_MODE_NONE) {
    int numdims = input->num_dims + 1;
    int dims[MAX_TENSOR_DIM];
    for (int i = 1; i < numdims; i++)
      dims[i] = input->dims[i - 1];
    dims[0] = out_dim;
    embed->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, embed->data_type, embed, 0, true /*create_grad*/);
  } else {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++)
      dims[i] = input->dims[i];
    dims[0] = out_dim;
    embed->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, embed->data_type, embed, 0, true /*create_grad*/);
  }
  {
    int dims[2] = {out_dim, num_entries};
    embed->weights[0] = create_weight_legion_ordering(2,
                                                      dims,
                                                      DT_FLOAT,
                                                      embed,
                                                      true /*create_grad*/,
                                                      kernel_initializer,
                                                      CHOSEN_SYNC_TYPE);
  }
  embed->data_type = DT_FLOAT;
  embed->add_int_property("num_entries", num_entries);
  embed->add_int_property("out_dim", out_dim);
  embed->add_int_property("aggr_mode", aggr);
  embed->add_initializer("kernel", kernel_initializer);
  layers.push_back(embed);
  return embed->outputs[0];
}

EmbeddingParams Embedding::get_params() const {
  EmbeddingParams params;
  params.num_entries = this->num_entries;
  params.out_channels = this->out_channels;
  params.aggr = this->aggr;
  // TODO: get rid of layer_guid
  // https://github.com/flexflow/FlexFlow/issues/304
  params.layer_guid = this->layer_guid;
  return params;
}

Op *Embedding::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("num_entries", value);
  int num_entries = value;
  layer->get_int_property("out_dim", value);
  int out_dim = value;
  layer->get_int_property("aggr_mode", value);
  AggrMode aggr = (AggrMode)value;
  Initializer *kernel_initializer;
  layer->get_initializer("kernel", kernel_initializer);
  return new Embedding(model,
                       layer->layer_guid,
                       inputs[0],
                       num_entries,
                       out_dim,
                       aggr,
                       false /*allocate_weights*/,
                       layer->name);
}

int Embedding::input_vocab_size_replica_dim() const {
  return this->inputs[0]->num_dims - 1;
}

int Embedding::input_channel_out_replica_dim() const {
  return this->inputs[0]->num_dims - 2;
}

int Embedding::output_vocab_size_replica_dim() const {
  assert(this->outputs[0] != nullptr);
  return this->outputs[0]->num_dims - 1;
}

int Embedding::output_size(ParallelDim output_dims[MAX_TENSOR_DIM]) {
  ParallelTensor const &input = this->inputs[0];

  int const OUT_CHANNELS = Output::OUT_CHANNELS;
  if (aggr == AGGR_MODE_NONE) {
    int num_dims = input->num_dims + 1;
    for (int i = 1; i < num_dims - 1; i++)
      output_dims[i] = input->dims[i - 1];
    assert(OUT_CHANNELS == 0);
    output_dims[OUT_CHANNELS].size = this->out_channels;
    output_dims[OUT_CHANNELS].degree = 1;
    output_dims[OUT_CHANNELS].parallel_idx = -1;
    // Currently do not support parallelizing over the replica dim
    output_dims[num_dims - 1].size = 1;
    output_dims[num_dims - 1].degree = 1;
    output_dims[num_dims - 1].parallel_idx = -1;
    output_dims[num_dims - 1].is_replica_dim = true;
    return num_dims;
  } else {
    int num_dims = input->num_dims;
    for (int i = 1; i < num_dims - 1; i++)
      output_dims[i] = input->dims[i];
    assert(OUT_CHANNELS == 0);
    output_dims[OUT_CHANNELS].size = this->out_channels;
    output_dims[OUT_CHANNELS].degree = 1;
    output_dims[OUT_CHANNELS].parallel_idx = -1;
    // Currently do not support parallelizing over the replica dim
    output_dims[num_dims - 1].size = 1;
    output_dims[num_dims - 1].degree = 1;
    output_dims[num_dims - 1].parallel_idx = -1;
    output_dims[num_dims - 1].is_replica_dim = true;
    return num_dims;
  }
  // const int REPLICA = this->output_vocab_size_replica_dim();
}

int Embedding::weight_size(ParallelDim weight_dims[MAX_TENSOR_DIM]) {
  ParallelTensor const &input = this->inputs[0];

  weight_dims[Weight::OUT_CHANNELS].size = this->out_channels;
  weight_dims[Weight::OUT_CHANNELS].degree = 1;
  weight_dims[Weight::OUT_CHANNELS].parallel_idx = -1;
  weight_dims[Weight::VOCAB_SIZE].size = this->num_entries;
  weight_dims[Weight::VOCAB_SIZE].degree = 1;
  weight_dims[Weight::VOCAB_SIZE].parallel_idx = -1;
  for (int i = 2; i < input->num_dims; i++) {
    weight_dims[i].size = input->dims[i - 1].degree;
    weight_dims[i].degree = weight_dims[i].size;
    weight_dims[i].parallel_idx = input->dims[i - 1].parallel_idx;
    weight_dims[i].is_replica_dim = true;
  }
  return input->num_dims;
}

void Embedding::register_output_mappings() {
  if (aggr == AGGR_MODE_NONE) {
    int num_dims = this->inputs[0]->num_dims + 1;
    for (int i = 1; i < num_dims - 1; i++)
      this->register_output_parallel_dims(i - 1, i);
  } else {
    int num_dims = this->inputs[0]->num_dims;
    for (int i = 1; i < num_dims - 1; i++)
      this->register_output_parallel_dims(i, i);
  }
#ifdef DEADCODE
  this->register_output_parallel_dims({
      {this->input_vocab_size_replica_dim(),
       this->output_vocab_size_replica_dim()},
      {this->input_channel_out_replica_dim(), Output::OUT_CHANNELS},
  });

  for (int i = 1; i < this->inputs[0]->num_dims - 1; i++) {
    this->register_output_parallel_dims(i - 1, i);
  }
#endif
}

void Embedding::register_weight_mappings() {
#ifdef DEADCODE
  this->register_weight_parallel_dims({
      {this->input_vocab_size_replica_dim(), Weight::VOCAB_SIZE},
      {this->input_channel_out_replica_dim(), Weight::OUT_CHANNELS},
  });
#endif

  for (int i = 2; i < this->inputs[0]->num_dims; i++) {
    this->register_weight_parallel_dims(i - 1, i);
  }
}

void Embedding::register_mappings() {
  this->register_output_mappings();
  this->register_weight_mappings();
}

/* Params */

bool EmbeddingParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

bool operator==(EmbeddingParams const &lhs, EmbeddingParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid &&
         lhs.out_channels == rhs.out_channels &&
         lhs.num_entries == rhs.num_entries && lhs.aggr == rhs.aggr;
}

Embedding::Embedding(FFModel &model,
                     EmbeddingParams const &params,
                     ParallelTensor const input,
                     bool allocate_weights,
                     char const *name)
    : Embedding(model,
                params.layer_guid,
                input,
                params.num_entries,
                params.out_channels,
                params.aggr,
                allocate_weights,
                name) {}

Embedding::Embedding(FFModel &model,
                     Embedding const &other,
                     const ParallelTensor input,
                     bool allocate_weights)
    : Embedding(model,
                other.layer_guid,
                input,
                other.num_entries,
                other.out_channels,
                other.aggr,
                allocate_weights,
                other.name) {}

Embedding::Embedding(FFModel &model,
                     LayerID const &_layer_guid,
                     const ParallelTensor _input,
                     int _num_entries,
                     int _out_channels,
                     AggrMode _aggr,
                     bool allocate_weights,
                     char const *name)
    : Op(model,
         OP_EMBEDDING,
         name,
         1 /*inputs*/,
         1 /*weights*/,
         allocate_weights,
         1 /*outputs*/,
         _input),
      num_entries(_num_entries), out_channels(_out_channels), aggr(_aggr) {
  layer_guid = _layer_guid;
  std::vector<ParallelDim *> weight_dim_sets;

  int weight_ndim;
  ParallelDim weight_dims[MAX_TENSOR_DIM];
  if (allocate_weights) {
    weight_ndim = this->weight_size(weight_dims);
    weight_dim_sets.push_back(weight_dims);
  }

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndim = this->output_size(output_dims);

  // register mappings between inputs/weights and outputs
  this->register_mappings();

  this->solve_parallel_dim_mappings(
      {_input->dims}, weight_dim_sets, {output_dims});

  if (allocate_weights) {
    Initializer *weight_initializer = new GlorotUniform(std::rand() /*seed*/);

    weights[0] =
        model.create_parallel_weight_legion_ordering(weight_ndim,
                                                     weight_dims,
                                                     DT_FLOAT,
                                                     nullptr /*owner_op*/,
                                                     true /*create_grad*/,
                                                     weight_initializer,
                                                     CHOSEN_SYNC_TYPE);
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_ndim, output_dims, DT_FLOAT, this);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Embedding::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(EMBED_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Embedding)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  // launcher.add_region_requirement(
  //  RegionRequirement(input_lps[0], 0/*projection*/,
  //    READ_ONLY, EXCLUSIVE, inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[3]: input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Embedding::init_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  Embedding const *embed = (Embedding *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  EmbeddingMeta *m = new EmbeddingMeta(handle);
  m->input_data_type = embed->inputs[0]->data_type;
  m->profiling = embed->profiling;
  m->aggr = embed->aggr;
  return m;
}

void Embedding::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(EMBED_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): kernel
*/
void Embedding::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  EmbeddingMeta const *m = *((EmbeddingMeta **)task->local_args);
  if (m->input_data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

template <typename TI>
void Embedding::forward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const Embedding* embed = (Embedding*) task->args;
  EmbeddingMeta const *m = *((EmbeddingMeta **)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain kernel_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  if (m->aggr == AGGR_MODE_NONE) {
    // assert(kernel_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_domain.get_dim());
    for (size_t i = 0; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_domain.hi()[i + 1]);
      assert(input_domain.lo()[i] == output_domain.lo()[i + 1]);
    }
    assert(kernel_domain.hi()[0] - kernel_domain.lo()[0] ==
           output_domain.hi()[0] - output_domain.lo()[0]);
  } else {
    // assert(kernel_domain.get_dim() == 2);
    assert(input_domain.get_dim() == output_domain.get_dim());
    for (size_t i = 1; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_domain.hi()[i]);
      assert(input_domain.lo()[i] == output_domain.lo()[i]);
    }
    assert(kernel_domain.hi()[0] - kernel_domain.lo()[0] ==
           output_domain.hi()[0] - output_domain.lo()[0]);
  }
  const TI *input_ptr = helperGetTensorPointerRO<TI>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *output_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *kernel_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int in_dim, out_dim, effective_batch_size;
  if (m->aggr == AGGR_MODE_NONE) {
    in_dim = 1;
    out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
    effective_batch_size = output_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  } else {
    in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
    out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
    effective_batch_size = output_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  }

  Embedding::forward_kernel_wrapper<TI>(m,
                                        input_ptr,
                                        output_ptr,
                                        kernel_ptr,
                                        in_dim,
                                        out_dim,
                                        effective_batch_size,
                                        m->aggr,
                                        output_domain.get_volume());
}

void Embedding::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(EMBED_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight_grad
  launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                    0 /*projection*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Embedding::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  EmbeddingMeta const *m = *((EmbeddingMeta **)task->local_args);
  if (m->input_data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

template <typename TI>
void Embedding::backward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const Embedding* embed = (Embedding*) task->args;
  EmbeddingMeta const *m = *((EmbeddingMeta **)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain kernel_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  if (m->aggr == AGGR_MODE_NONE) {
    // assert(kernel_grad_domain.get_dim() == 2);
    assert(input_domain.get_dim() + 1 == output_grad_domain.get_dim());
    for (size_t i = 0; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_grad_domain.hi()[i + 1]);
      assert(input_domain.lo()[i] == output_grad_domain.lo()[i + 1]);
    }
    assert(kernel_grad_domain.hi()[0] - kernel_grad_domain.lo()[0] ==
           output_grad_domain.hi()[0] - output_grad_domain.lo()[0]);
  } else {
    // assert(kernel_grad_domain.get_dim() == 2);
    assert(input_domain.get_dim() == output_grad_domain.get_dim());
    for (size_t i = 1; i < input_domain.get_dim(); i++) {
      assert(input_domain.hi()[i] == output_grad_domain.hi()[i]);
      assert(input_domain.lo()[i] == output_grad_domain.lo()[i]);
    }
    assert(kernel_grad_domain.hi()[0] - kernel_grad_domain.lo()[0] ==
           output_grad_domain.hi()[0] - output_grad_domain.lo()[0]);
  }
  const TI *input_ptr = helperGetTensorPointerRO<TI>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float const *output_grad_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float *kernel_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int in_dim, out_dim, effective_batch_size;
  if (m->aggr == AGGR_MODE_NONE) {
    in_dim = 1;
    out_dim = output_grad_domain.hi()[0] - output_grad_domain.lo()[0] + 1;
    effective_batch_size = output_grad_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  } else {
    in_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
    out_dim = output_grad_domain.hi()[0] - output_grad_domain.lo()[0] + 1;
    effective_batch_size = output_grad_domain.get_volume() / out_dim;
    assert(effective_batch_size * in_dim == input_domain.get_volume());
  }
  Embedding::backward_kernel_wrapper<TI>(m,
                                         input_ptr,
                                         output_grad_ptr,
                                         kernel_grad_ptr,
                                         in_dim,
                                         out_dim,
                                         effective_batch_size,
                                         m->aggr,
                                         output_grad_domain.get_volume());
}

bool Embedding::measure_operator_cost(Simulator *sim,
                                      MachineView const &mv,
                                      CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }

  EmbeddingMeta *m = sim->embedding_meta;
  assert(m->profiling == false);

  sim->free_all();
  bool out_of_memory = false;
  int64_t *input_ptr =
      (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  out_of_memory = out_of_memory || (input_ptr == NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *weight_ptr =
      (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
  cost_metrics.weights_memory += cost_metrics.total_mem_diff_from(sim->offset);
  out_of_memory = out_of_memory || (weight_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  int in_dim = aggr == AGGR_MODE_NONE ? 1 : sub_input.dims[0].size;
  int out_dim = sub_output.dims[0].size;
  int effective_batch_size = sub_output.get_volume() / out_dim;
  assert(effective_batch_size * in_dim == sub_input.get_volume());

  // Randomly initialize the intput tensor to avoid out of index range issues
  rand_generate_int64_wrapper(input_ptr, sub_input.get_volume(), num_entries);
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m,
                           input_ptr,
                           output_ptr,
                           weight_ptr,
                           in_dim,
                           out_dim,
                           effective_batch_size,
                           this->aggr,
                           sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *weight_grad_ptr =
        (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
    cost_metrics.weights_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);
    out_of_memory = out_of_memory || (weight_grad_ptr == NULL);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);
    out_of_memory = out_of_memory || (output_grad_ptr == NULL);

    int64_t *input_grad_ptr =
        (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
    out_of_memory = out_of_memory || (input_grad_ptr == NULL);

    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel_wrapper(m,
                              input_grad_ptr,
                              output_grad_ptr,
                              weight_grad_ptr,
                              in_dim,
                              out_dim,
                              effective_batch_size,
                              this->aggr,
                              sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf) "
           "backward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }

  return true;
}

void EmbeddingLookup_int64_t_float_float__avx2_fma(int const block_size,
                                                   int const output_size,
                                                   int const index_size,
                                                   int const data_size,
                                                   float const *input,
                                                   int64_t const *indices,
                                                   int const *lengths,
                                                   float const *weight,
                                                   bool normalize_by_lengths,
                                                   float *out) {
#ifdef FF_USE_AVX2
  const int64_t prefdist_T0 = 16;
  if (block_size == 128) {
    // unrolling 16 times
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float *op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      __m256 vop32 = _mm256_setzero_ps();
      __m256 vop40 = _mm256_setzero_ps();
      __m256 vop48 = _mm256_setzero_ps();
      __m256 vop56 = _mm256_setzero_ps();
      __m256 vop64 = _mm256_setzero_ps();
      __m256 vop72 = _mm256_setzero_ps();
      __m256 vop80 = _mm256_setzero_ps();
      __m256 vop88 = _mm256_setzero_ps();
      __m256 vop96 = _mm256_setzero_ps();
      __m256 vop104 = _mm256_setzero_ps();
      __m256 vop112 = _mm256_setzero_ps();
      __m256 vop120 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        float const *ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
                                    ? (dataInd + prefdist_T0)
                                    : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
               idx_pref_T0 < data_size);
        float const *ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
        vop32 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (32)), vop32);
        _mm_prefetch((&ip_next_T0[32]), _MM_HINT_T0);
        vop40 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (40)), vop40);
        _mm_prefetch((&ip_next_T0[40]), _MM_HINT_T0);
        vop48 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (48)), vop48);
        _mm_prefetch((&ip_next_T0[48]), _MM_HINT_T0);
        vop56 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (56)), vop56);
        _mm_prefetch((&ip_next_T0[56]), _MM_HINT_T0);
        vop64 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (64)), vop64);
        _mm_prefetch((&ip_next_T0[64]), _MM_HINT_T0);
        vop72 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (72)), vop72);
        _mm_prefetch((&ip_next_T0[72]), _MM_HINT_T0);
        vop80 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (80)), vop80);
        _mm_prefetch((&ip_next_T0[80]), _MM_HINT_T0);
        vop88 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (88)), vop88);
        _mm_prefetch((&ip_next_T0[88]), _MM_HINT_T0);
        vop96 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (96)), vop96);
        _mm_prefetch((&ip_next_T0[96]), _MM_HINT_T0);
        vop104 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (104)), vop104);
        _mm_prefetch((&ip_next_T0[104]), _MM_HINT_T0);
        vop112 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (112)), vop112);
        _mm_prefetch((&ip_next_T0[112]), _MM_HINT_T0);
        vop120 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (120)), vop120);
        _mm_prefetch((&ip_next_T0[120]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
        _mm256_storeu_ps(&op[32], vop32);
        _mm256_storeu_ps(&op[40], vop40);
        _mm256_storeu_ps(&op[48], vop48);
        _mm256_storeu_ps(&op[56], vop56);
        _mm256_storeu_ps(&op[64], vop64);
        _mm256_storeu_ps(&op[72], vop72);
        _mm256_storeu_ps(&op[80], vop80);
        _mm256_storeu_ps(&op[88], vop88);
        _mm256_storeu_ps(&op[96], vop96);
        _mm256_storeu_ps(&op[104], vop104);
        _mm256_storeu_ps(&op[112], vop112);
        _mm256_storeu_ps(&op[120], vop120);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
        _mm256_storeu_ps(&op[32], _mm256_mul_ps(vop32, vlen_inv));
        _mm256_storeu_ps(&op[40], _mm256_mul_ps(vop40, vlen_inv));
        _mm256_storeu_ps(&op[48], _mm256_mul_ps(vop48, vlen_inv));
        _mm256_storeu_ps(&op[56], _mm256_mul_ps(vop56, vlen_inv));
        _mm256_storeu_ps(&op[64], _mm256_mul_ps(vop64, vlen_inv));
        _mm256_storeu_ps(&op[72], _mm256_mul_ps(vop72, vlen_inv));
        _mm256_storeu_ps(&op[80], _mm256_mul_ps(vop80, vlen_inv));
        _mm256_storeu_ps(&op[88], _mm256_mul_ps(vop88, vlen_inv));
        _mm256_storeu_ps(&op[96], _mm256_mul_ps(vop96, vlen_inv));
        _mm256_storeu_ps(&op[104], _mm256_mul_ps(vop104, vlen_inv));
        _mm256_storeu_ps(&op[112], _mm256_mul_ps(vop112, vlen_inv));
        _mm256_storeu_ps(&op[120], _mm256_mul_ps(vop120, vlen_inv));
      }
    }
    __m256 vwgt = _mm256_set1_ps(wgt);
    float const *ip = &input[idx * block_size];
    const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
                                ? (dataInd + prefdist_T0)
                                : dataInd;
    const int64_t idx_pref_T0 = indices[next_T0];
    assert(idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
           idx_pref_T0 < data_size);
    float const *ip_next_T0 = &input[idx_pref_T0 * block_size];
    vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
    _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
    vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
    _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
    vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
    _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
    vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
    _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
  }
  if (normalize_by_lengths == false) {
    _mm256_storeu_ps(&op[0], vop0);
    _mm256_storeu_ps(&op[8], vop8);
    _mm256_storeu_ps(&op[16], vop16);
    _mm256_storeu_ps(&op[24], vop24);
  } else if (lengths[rangeIndex]) {
    __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
    _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
    _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
    _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
    _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
  }
}
}
else {
  // generic code
  int64_t dataInd = 0;
  for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
    float *op = &out[rangeIndex * block_size];
    int j = 0;
    for (; j + 8 <= block_size; j += 8) {
      _mm256_storeu_ps(op + j, _mm256_setzero_ps());
    }
    for (; j < block_size; j++) {
      op[j] = 0.0f;
    }
    for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
         ++dataInd) {
      const int64_t idx = indices[dataInd];
      float wgt = 1.f;
      if (weight) {
        wgt = weight[dataInd];
      }
      __m256 vwgt = _mm256_set1_ps(wgt);
      float const *ip = &input[idx * block_size];
      const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
                                  ? (dataInd + prefdist_T0)
                                  : dataInd;
      const int64_t idx_pref_T0 = indices[next_T0];
      assert(idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
             idx_pref_T0 < data_size);
      float const *ip_next_T0 = &input[idx_pref_T0 * block_size];
      j = 0;
      for (; j + 8 <= block_size; j += 8) {
        _mm256_storeu_ps(&op[j],
                         _mm256_fmadd_ps(vwgt,
                                         _mm256_loadu_ps(&ip[j]),
                                         _mm256_loadu_ps(&op[j])));
        _mm_prefetch((&ip_next_T0[j]), _MM_HINT_T0);
      }
      for (; j < block_size; j++) {
        op[j] += wgt * ip[j];
      }
    }
    if (normalize_by_lengths && lengths[rangeIndex]) {
      float len_inv = 1.0f / lengths[rangeIndex];
      __m256 vlen_inv = _mm256_set1_ps(len_inv);
      j = 0;
      for (; j + 8 <= block_size; j += 8) {
        _mm256_storeu_ps(&op[j],
                         _mm256_mul_ps(_mm256_loadu_ps(&op[j]), vlen_inv));
      }
      for (; j < block_size; j++) {
        op[j] = len_inv * op[j];
      }
    }
  }
}
#else
  assert(0);
#endif
}

void embed_forward(int64_t const *input,
                   int const *lengths,
                   float *output,
                   float const *embed,
                   int block_size,
                   int output_size,
                   int index_size,
                   int data_size) {
  EmbeddingLookup_int64_t_float_float__avx2_fma(block_size,
                                                output_size,
                                                index_size,
                                                data_size,
                                                embed,
                                                input,
                                                lengths,
                                                nullptr,
                                                false,
                                                output);
}

void embed_backward_generic(int64_t const *input,
                            int const *lengths,
                            float const *output,
                            float *embed,
                            int block_size,
                            int output_size,
                            int index_size,
                            int data_size) {
  // FIXME: Not functionaly correct.
  for (int i = 0; i < output_size * block_size; i++) {
    int idx = i / block_size;
    int off = i % block_size;
    int64_t wordIdx = input[idx];
    // FIXME: Need to be atomic depending on the strategy
    embed[wordIdx * block_size + off] += output[i];
    ;
  }
}

void embed_backward(int64_t const *input,
                    int const *lengths,
                    float const *output,
                    float *embed,
                    int block_size,
                    int output_size,
                    int index_size,
                    int data_size) {
  embed_backward_generic(input,
                         lengths,
                         output,
                         embed,
                         block_size,
                         output_size,
                         index_size,
                         data_size);
}

void Embedding::forward_task_cpu(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const Embedding* embed = (Embedding*) task->args;
  AccessorRO<int64_t, 2> const acc_input(regions[0], FID_DATA);
  AccessorWO<float, 2> const acc_output(regions[1], FID_DATA);
  AccessorRO<float, 2> const acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  // const int64_t* input = acc_input.ptr(rect_input);
  // float* output = acc_output.ptr(rect_output);
  // const float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int data_size = 1000000; // FIXME
  // For now we are assuming the length is always 1
  int index_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(in_dim == 1);
  std::vector<int> lengths(output_size, 1);
  embed_forward(acc_input.ptr(rect_input),
                lengths.data(),
                acc_output.ptr(rect_output),
                acc_weight.ptr(rect_weight),
                block_size,
                output_size,
                index_size,
                data_size);
}

void Embedding::backward_task_cpu(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const Embedding* embed = (Embedding*) task->args;
  AccessorRO<int64_t, 2> const acc_input(regions[0], FID_DATA);
  AccessorRO<float, 2> const acc_output(regions[1], FID_DATA);
  AccessorRW<float, 2> const acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  // coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  // const int64_t* input = acc_input.ptr(rect_input);
  // const float* output = acc_output.ptr(rect_output);
  // float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int index_size = rect_input.hi[1] - rect_input.lo[0] + 1;
  int data_size = 1000000; // FIXME
  std::vector<int> lengths(output_size, 1);
  embed_backward(acc_input.ptr(rect_input),
                 lengths.data(),
                 acc_output.ptr(rect_output),
                 acc_weight.ptr(rect_weight),
                 block_size,
                 output_size,
                 index_size,
                 data_size);
}
}
; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::EmbeddingParams>::operator()(
    FlexFlow::EmbeddingParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.out_channels);
  hash_combine(key, params.aggr);
  hash_combine(key, params.num_entries);
  return key;
}
}; // namespace std
