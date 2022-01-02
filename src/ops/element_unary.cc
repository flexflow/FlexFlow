#include "flexflow/ops/element_unary.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::unary(OperatorType op,
                      const Tensor x,
                      bool inplace,
                      const char *name,
                      float scalar)
{
  Layer *ele = new Layer(this, op, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, x);
  int numdims = x->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++)
    dims[i] = x->dims[i];
  ele->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT,
                                                  ele, 0, true/*create_grad*/);
  ele->add_int_property("inplace", inplace);
  ele->add_float_property("scalar", scalar);
  layers.push_back(ele);
  return ele->outputs[0];
}

Op* ElementUnary::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("inplace", value);
  bool inplace = (bool) value;
  float scalar;
  layer->get_float_property("scalar", scalar);
  return new ElementUnary(model, layer->op_type, inputs[0], inplace,
      layer->name, scalar);
}

size_t ElementUnary::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->op_type);
  hash_combine(hash, this->inplace);
  if (this->op_type == OP_SCALAR_MULTIPLY) {
    hash_combine(hash, this->scalar);
  }
  return hash;
}

using PCG::Node;
Node FFModel::get_or_create_element_unary_node(const ParallelTensor input,
                                               OperatorType op,
                                               bool inplace,
                                               float scalar)
{
  if (input->dims[input->num_dims-1].degree != 1) {
    return Node::INVALID_NODE;
  }

  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, op);
  hash_combine(hash, inplace);
  if (op == OP_SCALAR_MULTIPLY) {
    hash_combine(hash, scalar);
  }

  ElementUnary *unary;
  const auto &it = this->cached_element_unary_ops.find(hash);
  if (it != cached_element_unary_ops.end()) { 
    unary = it->second;
  } else {
    unary = new ElementUnary(*this, op, input, inplace, NULL, scalar);
    cached_element_unary_ops[hash] = unary;
  }

  return this->new_node(unary);
}

Tensor FFModel::exp(const Tensor x,
                    const char *name)
{
  return this->unary(OP_EXP, x, false/*inplace*/, name);
}

Tensor FFModel::scalar_multiply(const Tensor x, const float scalar, bool inplace, const char *name)
{
  return this->unary(OP_SCALAR_MULTIPLY, x, inplace, name, scalar);
}

Tensor FFModel::scalar_add(const Tensor x,const float scalar ,bool inplace, const char *name)
{
  return this->unary(OP_SCALAR_ADD, x, inplace, name, scalar);
}

Tensor FFModel::scalar_sub(const Tensor x,const float scalar ,bool inplace, const char *name)
{
  return this->unary(OP_SCALAR_SUB, x, inplace, name, scalar);
}

Tensor FFModel::scalar_truediv(const Tensor x,const float scalar ,bool inplace, const char *name)
{
  return this->unary(OP_SCALAR_TRUE_DIV, x, inplace, name, scalar);
}

Tensor FFModel::relu(const Tensor x, bool inplace, const char *name)
{
  return this->unary(OP_RELU, x, inplace, name);
}

Tensor FFModel::sigmoid(const Tensor x, const char *name)
{
  return this->unary(OP_SIGMOID, x, false/*inplace*/, name);
}

Tensor FFModel::tanh(const Tensor x, const char *name)
{
  return this->unary(OP_TANH, x, false/*inplace*/, name);
}

Tensor FFModel::identity(const Tensor x, const char *name)
{
  return this->unary(OP_IDENTITY, x, false/*inplace*/, name);
}

Tensor FFModel::gelu(const Tensor x, const char *name)
{
  return this->unary(OP_GELU, x, false/*inplace*/, name);
}

Tensor FFModel::elu(const Tensor x, bool inplace, const char *name)
{
  // Currently assume inplace is false
  assert(!inplace);
  return this->unary(OP_ELU, x, inplace, name);
}

ElementUnary::ElementUnary(FFModel& model,
                           OperatorType _op_type,
                           const ParallelTensor x,
                           bool _inplace,
                           const char* name,
                           float _scalar)
: Op(model, _op_type, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, x), inplace(_inplace), scalar(_scalar)
{
  numOutputs = 1;
  int numdim = x->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = x->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(numdim, dims, x->data_type, this);
}

void ElementUnary::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(ELEMENTUNARY_INIT_TASK_ID, parallel_is,
                              TaskArgument(this, sizeof(ElementUnary)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  assert (!inplace);
  if (!inplace) {
    init_launcher.add_region_requirement(
        RegionRequirement(outputs[0]->part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    init_launcher.add_field(1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void ElementUnary::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(0, FID_DATA);
  } else {
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
         WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void ElementUnary::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  if (inplace) {
    assert(inputs[0]->part == outputs[0]->part);
    assert(inputs[0]->part_grad == outputs[0]->part_grad);
    // regions[2](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
  } else {
    // regions[0](I): input
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[1](I/O): input_grad
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
    // regions[2](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(2, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void ElementUnary::serialize(Legion::Serializer& sez) const {
  sez.serialize(this->op_type);
  sez.serialize(this->inplace);
  if (this->op_type == OP_SCALAR_MULTIPLY) {
    sez.serialize(scalar);
  }
}

using PCG::Node;
/*static*/
Node ElementUnary::deserialize(FFModel& ff, Legion::Deserializer& dez, ParallelTensor inputs[], int num_inputs) {
  assert (num_inputs == 1);
  OperatorType op_type;
  float scalar;
  bool inplace;
  dez.deserialize(op_type);
  dez.deserialize(inplace);
  if (op_type == OP_SCALAR_MULTIPLY) {
    dez.deserialize(scalar);
  }

  return ff.get_or_create_element_unary_node(inputs[0], op_type, inplace, scalar);
}

Op *ElementUnary::materialize(FFModel& ff, ParallelTensor inputs[], int num_inputs) const {
  assert (num_inputs == 1);
  return new ElementUnary(ff, this->op_type, inputs[0], this->inplace, this->name, this->scalar);
}

}; // namespace FlexFlow
