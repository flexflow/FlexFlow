#include "flexflow/ops/lora_linear.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/lora_linear_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

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

using namespace FlexFlow::Kernels::LoraLinear;

void FFModel::lora_linear(Tensor const input,
                          Tensor const output,
                          int rank,
                          DataType data_type,
                          Initializer *kernel_initializer,
                          char const *name) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  assert(data_type == input->data_type);
  assert(data_type == output->data_type);
  Layer *li = nullptr;
  li = new Layer(this,
                 OP_LORA_LINEAR,
                 data_type,
                 name,
                 2 /*inputs*/,
                 2 /*weights*/,
                 1 /*outputs*/,
                 input,
                 output);
  {
    int numdims = output->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = output->dims[i];
    }
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, data_type, li, 0, true /*create_grad*/);
  }
  {
    int dims[2] = {input->dims[0], rank};
    li->weights[0] = create_weight_legion_ordering(2,
                                                   dims,
                                                   data_type,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
  }
  {
    int dims[2] = {rank, output->dims[0]};
    li->weights[1] = create_weight_legion_ordering(2,
                                                   dims,
                                                   data_type,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
  }
  li->add_int_property("rank", rank);
  layers.push_back(li);
}

Op *LoraLinear::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("rank", value);
  int rank = (int)value;
  return new LoraLinear(model,
                        layer->layer_guid,
                        inputs[0],
                        inputs[1],
                        rank,
                        layer->data_type,
                        false /*allocate_weights*/,
                        layer->name);
}

LoraLinear::LoraLinear(FFModel &model,
                       LoraLinear const &other,
                       ParallelTensor const input,
                       ParallelTensor const output,
                       bool allocate_weights)
    : LoraLinear(model,
                 other.layer_guid,
                 input,
                 output,
                 other.rank,
                 other.data_type,
                 allocate_weights,
                 other.name) {}

LoraLinear::LoraLinear(FFModel &model,
                       Params const &params,
                       Input const &inputs,
                       bool allocate_weights,
                       char const *name)
    : LoraLinear(model,
                 params.layer_guid,
                 inputs.first,
                 inputs.second,
                 params.rank,
                 params.data_type,
                 allocate_weights,
                 name) {}

LoraLinear::LoraLinear(FFModel &model,
                       LayerID const &_layer_guid,
                       ParallelTensor const _input,
                       ParallelTensor const _output,
                       int _rank,
                       DataType _data_type,
                       bool allocate_weights,
                       char const *name)
    : Op(model,
         OP_LORA_LINEAR,
         _data_type,
         name,
         2 /*inputs*/,
         2 /*weights*/,
         allocate_weights,
         1 /*outputs*/,
         _input,
         _output),
      rank(_rank) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  data_type = _data_type;

  ParallelTensorShape input_shape = this->inputs[0]->get_shape();
  LoraLinearParams params = this->get_params();

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
    // create weight first
    {
      ParallelDim dims[2];
      int num_dims = inputs[0]->num_dims;
      dims[1] = inputs[0]->dims[num_dims - 1]; // data parallel
      dims[1].size = dims[1].degree;
      dims[1].is_replica_dim = true;
      dims[0] = inputs[0]->dims[0];
      dims[0].size = inputs[0]->dims[0].size * rank;
      weights[0] =
          model.create_parallel_weight_legion_ordering(2,
                                                       dims,
                                                       this->data_type,
                                                       nullptr /*owner_op*/,
                                                       true /*create_grad*/,
                                                       kernel_initializer,
                                                       CHOSEN_SYNC_TYPE);
    }
    // create weight second
    {
      ParallelDim dims[2];
      int num_dims = inputs[0]->num_dims;
      dims[1] = inputs[0]->dims[0];
      dims[1].size = dims[1].degree;
      dims[1].is_replica_dim = true;
      dims[0] = inputs[1]->dims[0];
      dims[0].size = inputs[1]->dims[0].size * rank;
      weights[1] =
          model.create_parallel_weight_legion_ordering(2,
                                                       dims,
                                                       this->data_type,
                                                       nullptr /*owner_op*/,
                                                       true /*create_grad*/,
                                                       kernel_initializer,
                                                       CHOSEN_SYNC_TYPE);
    }
  }
  // Create output tensor
  {
    int numdim = inputs[1]->num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdim; i++) {
      dims[i] = inputs[1]->dims[i];
    }
    outputs[0] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, inputs[1]->data_type, this);
  }
  // assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void LoraLinear::init(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal init");
}

void LoraLinear::init_inference(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output is the same as the second input
  assert(batch_outputs[0] == batch_inputs[1]);
  // assert(check_output_input_weight_same_machine_view());
  // output is considered as an input to allow in-place optimization
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, output_tensor);
  IndexLauncher launcher(LORA_LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(LoraLinear)),
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
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[1]->region));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, output_tensor);
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta *LoraLinear::init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  LoraLinear const *lora = (LoraLinear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  GenericTensorAccessorR input =
      helperGetGenericTensorAccessorRO(lora->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorRW(lora->outputs[0]->data_type,
                                       regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW weight_first =
      helperGetGenericTensorAccessorRW(lora->weights[0]->data_type,
                                       regions[2],
                                       task->regions[2],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW weight_second =
      helperGetGenericTensorAccessorRW(lora->weights[1]->data_type,
                                       regions[3],
                                       task->regions[3],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  int rank = lora->rank;
  int batch_size = output.domain.get_volume() / out_dim;
  assert(input.domain.get_volume() == in_dim * batch_size);
  assert(weight_first.domain.get_volume() == in_dim * rank);
  assert(weight_second.domain.get_volume() == out_dim * rank);

  LoraLinearMeta *m = new LoraLinearMeta(handle, lora);
  m->trainable_inputs[0] = lora->trainable_inputs[0];
  std::strcpy(m->op_name, lora->name);

  return m;
}

void LoraLinear::forward(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal init");
}

FutureMap
    LoraLinear::inference(FFModel const &ff,
                          BatchConfigFuture const &bc,
                          std::vector<ParallelTensor> const &batch_inputs,
                          std::vector<ParallelTensor> const &batch_outputs,
                          MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output is the same as the second input
  assert(batch_outputs[0] == batch_inputs[1]);
  // assert(check_output_input_weight_same_machine_view());
  // output is considered as an input to allow in-place optimization
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_inference(ff, argmap, output_tensor);
  IndexLauncher launcher(LORA_LINEAR_INF_TASK_ID,
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
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[1]->region));
  launcher.add_field(3, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void LoraLinear::inference_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  LoraLinearMeta *m = *((LoraLinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }
  assert(regions.size() == 4);
  assert(task->regions.size() == regions.size());
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorRW(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight_first = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight_second = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[3], task->regions[3], FID_DATA, ctx, runtime);
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  int rank = weight_first.domain.get_volume() / in_dim;
  assert(in_dim * rank == weight_first.domain.get_volume());
  assert(out_dim * rank == weight_second.domain.get_volume());

  int num_infr_tokens = bc->num_active_infr_tokens();
  int num_peft_tokens = bc->num_active_peft_tokens();
  inference_kernel_wrapper(m,
                           input.ptr,
                           output.ptr,
                           weight_first.ptr,
                           weight_second.ptr,
                           in_dim,
                           out_dim,
                           rank,
                           num_infr_tokens,
                           num_peft_tokens);
}

FutureMap LoraLinear::peft_bwd(FFModel const &ff,
                               BatchConfigFuture const &bc,
                               std::vector<ParallelTensor> const &batch_inputs,
                               std::vector<ParallelTensor> const &batch_outputs,
                               MachineView const *mv) {
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output is the same as the second input
  assert(batch_outputs[0] == batch_inputs[1]);
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  set_argumentmap_for_inference(ff, argmap, output_tensor);
  size_t machine_view_hash = view->hash();
  IndexLauncher launcher(LORA_LINEAR_PEFT_BWD_TASK_ID,
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
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[1]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region_grad));
  launcher.add_field(4, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[1]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[1]->region_grad));
  launcher.add_field(5, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void LoraLinear::peft_bwd_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LoraLinearMeta *m = *((LoraLinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  assert(regions.size() == 6);
  assert(task->regions.size() == regions.size());
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output_grad = helperGetGenericTensorAccessorRW(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight_first = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight_second = helperGetGenericTensorAccessorRO(
      m->weight_type[1], regions[3], task->regions[3], FID_DATA, ctx, runtime);
  GenericTensorAccessorW weight_first_grad = helperGetGenericTensorAccessorRW(
      m->weight_type[0], regions[4], task->regions[4], FID_DATA, ctx, runtime);
  GenericTensorAccessorW weight_second_grad = helperGetGenericTensorAccessorRW(
      m->weight_type[1], regions[5], task->regions[5], FID_DATA, ctx, runtime);

  int in_dim = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  int out_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  int rank = weight_first.domain.get_volume() / in_dim;
  assert(in_dim * rank == weight_first.domain.get_volume());
  assert(out_dim * rank == weight_second.domain.get_volume());
  assert(weight_first.domain == weight_first_grad.domain);
  assert(weight_second.domain == weight_second_grad.domain);

  int num_infr_tokens = bc->num_active_infr_tokens();
  int num_peft_tokens = bc->num_active_peft_tokens();
  peft_bwd_kernel_wrapper(m,
                          input_grad.ptr,
                          output_grad.ptr,
                          weight_first.ptr,
                          weight_second.ptr,
                          weight_first_grad.ptr,
                          weight_second_grad.ptr,
                          in_dim,
                          out_dim,
                          rank,
                          num_infr_tokens,
                          num_peft_tokens);
}

void LoraLinear::backward(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal backward");
}

void LoraLinear::print_layer(FFModel const &ff) {}

void LoraLinear::map_output_tensors(FFModel &ff) {
  assert(numOutputs == 1);
  assert(numInputs == 2);
  assert(outputs[0]->get_volume() == inputs[1]->get_volume());
  outputs[0]->parallel_is = inputs[1]->parallel_is;
  outputs[0]->region = inputs[1]->region;
  outputs[0]->part = inputs[1]->part;
  outputs[0]->region_grad = inputs[1]->region_grad;
  outputs[0]->part_grad = inputs[1]->part_grad;
}

bool LoraLinear::measure_operator_cost(Simulator *sim,
                                       MachineView const &mv,
                                       CostMetrics &cost_metrics) const {
  return false;
}

bool operator==(LoraLinearParams const &lhs, LoraLinearParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.rank == rhs.rank &&
         lhs.data_type == rhs.data_type;
}

void LoraLinear::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->rank);
  sez.serialize(this->data_type);
}

/* static */
using PCG::Node;
Node LoraLinear::deserialize(FFModel &ff,
                             Legion::Deserializer &dez,
                             ParallelTensor inputs[],
                             int num_inputs) {
  assert(num_inputs == 2);
  int rank;
  DataType data_type;
  size_t id, transformer_layer_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  LayerID layer_guid(id, transformer_layer_id);
  dez.deserialize(rank);
  dez.deserialize(data_type);

  LoraLinearParams params;
  params.rank = rank;
  params.data_type = data_type;
  params.layer_guid = layer_guid;
  return ff.get_or_create_node<LoraLinear>({inputs[0], inputs[1]}, params);
}

Op *LoraLinear::materialize(FFModel &ff,
                            ParallelTensor inputs[],
                            int num_inputs) const {
  LoraLinearParams params = get_params();
  return new LoraLinear(ff, params, {inputs[0], inputs[1]}, this->name);
}

LoraLinearParams LoraLinear::get_params() const {
  LoraLinearParams params;
  params.layer_guid = this->layer_guid;
  params.rank = this->rank;
  params.data_type = this->data_type;
  return params;
}

bool LoraLinearParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input_shape)
    const {
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::LoraLinearParams>::operator()(
    FlexFlow::LoraLinearParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.rank);
  hash_combine(key, params.data_type);
  return key;
}
}; // namespace std
