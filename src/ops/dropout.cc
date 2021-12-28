#include "flexflow/ops/dropout.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

Tensor FFModel::dropout(const Tensor input,
                        float rate,
                        unsigned long long seed,
                        const char* name)
{
  Layer* dropout = new Layer(this, OP_DROPOUT, name, 1/*inputs*/,
                             0/*weights*/, 1/*outputs*/, input);
  int numdims = input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++)
    dims[i] = input->dims[i];
  dropout->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT,
                                                      dropout, 0, true/*create_grad*/);
  dropout->add_float_property("rate", rate);
  dropout->add_int_property("seed", seed);
  layers.push_back(dropout);
  return dropout->outputs[0];
#ifdef DEADCODE
  // see = 0 is preserved as None, so we use a random seed
  if (seed == 0) {
    seed = std::rand();
  }
  Dropout *dropout = new Dropout(*this, input, rate, seed, name);
  layers.push_back(dropout);
  return dropout->outputs[0];
#endif
}

Op* Dropout::create_operator_from_layer(
        FFModel& model,
        const Layer* layer,
        const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("seed", value);
  int seed = value;
  float rate;
  layer->get_float_property("rate", rate);
  return new Dropout(model, 
      inputs[0],
      rate,
      seed,
      layer->name);
}

DropoutParams Dropout::get_params() const {
  DropoutParams params;
  params.rate = this->rate;
  params.seed = this->seed;

  return params;
}

size_t DropoutParams::get_hash(const ParallelTensor input) const {
  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, this->rate);
  hash_combine(hash, this->seed);

  return hash;
}

size_t Dropout::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

using PCG::Node;
Node FFModel::get_or_create_dropout_node(const ParallelTensor input,
                                         const DropoutParams& params)
{
  // Don't check is_valid since all inputs should be valid for dropout
  //if (!params.is_valid(input)) {
  //  return Node::INVALID_NODE;
  //}

  size_t hash = params.get_hash(input);

  Dropout *dropout = nullptr;
  const auto &it = this->cached_dropout_ops.find(hash);
  if (it != this->cached_dropout_ops.end()) {
    dropout = it->second;
  } else {
    dropout = new Dropout(*this, input, params.rate, params.seed, nullptr);
    cached_dropout_ops[hash] = dropout;
  }

  return this->new_node(dropout);
}

Dropout::Dropout(FFModel& model,
                 const ParallelTensor _input,
                 float _rate,
                 unsigned long long _seed,
                 const char* name)
: Op(model, OP_DROPOUT, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input),
  rate(_rate), seed(_seed)
{
  // Set output shape
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input->num_dims; i++)
    dims[i] = _input->dims[i];
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(_input->num_dims, dims, DT_FLOAT, this);
}

Dropout::Dropout(FFModel& model,
                 Dropout const &other,
                 const ParallelTensor input)
: Dropout(model, input, other.rate, other.seed, other.name)
{}

#ifdef DEADCODE
void Dropout::map_output_tensors(FFModel& model)
{
  int dim = inputs[0].num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      task_is = model.get_or_create_task_is(DIM, name); \
      map_output_tensors_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      assert(false && "Unsupported dim");
    }
  }
}

template<int NDIM>
void Dropout::map_output_tensors_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  Rect<NDIM> input_rect;
  input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[0]->part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
}
#endif

void Dropout::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(DROPOUT_INIT_TASK_ID, parallel_is,
                              TaskArgument(this, sizeof(Dropout)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Dropout::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(DROPOUT_FWD_TASK_ID, parallel_is,
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

void Dropout::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(DROPOUT_BWD_TASK_ID, parallel_is,
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

void Dropout::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->rate);
  sez.serialize(this->seed);
}

Node Dropout::deserialize(FFModel& ff, Legion::Deserializer& dez, ParallelTensor inputs[], int num_inputs) {
  assert (num_inputs == 1);
  unsigned long long seed;
  float rate;
  dez.deserialize(rate);
  dez.deserialize(seed);
  DropoutParams params;
  params.rate = rate;
  params.seed = seed;
  return ff.get_or_create_dropout_node(inputs[0], params);
}

}; // namespace FlexFlow
