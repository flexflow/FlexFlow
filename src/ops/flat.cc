#include "flexflow/ops/flat.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::flat(const Tensor input,
                     const char* name)
{
  assert(input->num_dims == 4);
  Layer *flat = new Layer(this, OP_FLAT, name, 1/*inputs*/, 0/*weights*/,
                          1/*outputs*/, input);
  int dims[MAX_TENSOR_DIM];
  dims[1] = input->dims[3];
  dims[0] = input->dims[2] * input->dims[1] * input->dims[0];
  flat->outputs[0] = create_tensor_legion_ordering(2, dims, DT_FLOAT,
                                                   flat, 0, true/*create_grad*/);
  layers.push_back(flat);
  return flat->outputs[0];
#ifdef DEADCODE
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this, input, name);
  layers.push_back(flat);
  return flat->outputs[0];
#endif
}

Op* Flat::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  return new Flat(model, inputs[0], layer->name);
}

int output_size(const ParallelTensor input, ParallelDim output_dims[MAX_TENSOR_DIM]) {
  output_dims[Output::REPLICA].is_replica_dim = true;
  output_dims[Output::SAMPLE].size = input->dims[Input::SAMPLE].size;
  output_dims[Output::CHANNEL].size = (
      input->dims[Input::CHANNEL].size * input->dims[Input::HEIGHT].size * input->dims[Input::WIDTH].size 
  );

  return Output::NUMDIM;
}


void solve_dims(const ParallelTensor input, 
                ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims) 
{
  assert ((output_dims == nullptr) == (output_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Flat::construct_output_mappings(mapping);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    *output_ndims = output_size(input, output_dims);
    output_dim_sets.push_back(output_dims);
  }

  solve_parallel_dim_mappings(
      mapping,
      {input->dims},
      {},
      output_dim_sets
  );
}

bool is_valid(const ParallelTensor input) 
{
  ParallelTensorShape output_shape;

  solve_dims(
      input,
      output_shape.dims, &output_shape.num_dims
  );

  bool is_valid = true;
  is_valid &= input->check_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= (input->dims[Input::WIDTH].degree == 1);
  is_valid &= (input->dims[Input::WIDTH].degree == 1);

  return is_valid;
}

size_t Flat::get_params_hash() const {
  return this->inputs[0]->get_owner_independent_hash();
}

using PCG::Node;
Node FFModel::get_or_create_flat_node(const ParallelTensor input) 
{
  if (!is_valid(input)) {
    return Node::INVALID_NODE;
  }

  size_t hash = input->get_owner_independent_hash();

  Flat *flat;

  const auto &it = this->cached_flat_ops.find(hash);
  if (it != cached_flat_ops.end()) {
    flat = it->second;
  } else {
    flat = new Flat(*this, input, NULL);
    cached_flat_ops[hash] = flat;
  }

  return this->new_node(flat);
}

/*static*/
void Flat::construct_output_mappings(std::vector<ParallelDimMappingRecord>& mappings) {
  Op::construct_output_parallel_dims(
    mappings,
    {
      { Input::REPLICA, MappingOperation::PARTITION, Output::REPLICA },
      { Input::SAMPLE, MappingOperation::PARTITION, Output::SAMPLE },
      { Input::CHANNEL, MappingOperation::PARTITION, Output::CHANNEL }
    }
  );
}

Flat::Flat(FFModel& model,
           const ParallelTensor _input,
           const char* name)
: Op(model, OP_FLAT, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input)
{
  assert(_input->num_dims == Input::NUMDIM);

  Flat::construct_output_mappings(*this->parallel_dims_mapping);

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;
  solve_dims(
      this->inputs[0],
      output_dims, &output_ndims
  );

  outputs[0] = model.create_parallel_tensor_legion_ordering(output_ndims, output_dims, _input->data_type, this);

  assert(check_output_input_weight_parallel_dims());
}

void Flat::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(FLAT_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Flat)), argmap,
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

void Flat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(FLAT_FWD_TASK_ID, parallel_is,
    TaskArgument(NULL, 0), argmap,
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

void Flat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(FLAT_BWD_TASK_ID, parallel_is,
    TaskArgument(NULL, 0), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

Domain Flat::get_input_tensor_shape(const ParallelConfig& pc,
    int input_idx, int part_idx) const
{
  assert(input_idx < numInputs);
  assert(pc.nDims == 3);
  assert(pc.dim[0] == 1);
  assert(pc.dim[2] == 1);

  Domain d;
  d.dim = inputs[input_idx]->num_dims;
  for (int i = 0; i < d.dim-1; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = inputs[input_idx]->dims[i].size - 1;
  }
  assert(inputs[input_idx]->dims[d.dim-2].size % pc.num_parts() == 0);
  int dim_size = inputs[input_idx]->dims[d.dim-2].size / pc.num_parts();
  d.rect_data[d.dim-2] = part_idx * dim_size;
  d.rect_data[2*d.dim-2] = d.rect_data[d.dim-2] + dim_size - 1;
  return d;
}

void Flat::serialize(Legion::Serializer& sez) const {
  return; 
}

using PCG::Node;
/*static*/
Node Flat::deserialize(FFModel& ff, Legion::Deserializer& dez, ParallelTensor inputs[], int num_inputs) {
  assert (num_inputs == 1);
  return ff.get_or_create_flat_node(inputs[0]);
}

Op *Flat::materialize(FFModel& ff, ParallelTensor inputs[], int num_inputs) const {
  assert (num_inputs == 1);
  return new Flat(ff, inputs[0], this->name);
}

}; // namespace FlexFlow
