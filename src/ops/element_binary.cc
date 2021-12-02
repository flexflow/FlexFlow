#include "flexflow/ops/element_binary.h"
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

Tensor FFModel::binary(OperatorType op,
                       const Tensor in1,
                       const Tensor in2,
                       bool inplace_a,
                       char const *name)
{
  Layer *ele = new Layer(this, op, name, 2/*inputs*/, 0/*weights*/, 1/*outputs*/, in1, in2);
  assert(in1->num_dims == in2->num_dims);
  ele->outputs[0]->num_dims = in1->num_dims;
  for (int i = 0; i < in1->num_dims; i++) {
    assert(in1->dims[i] == in2->dims[i]);
    ele->outputs[0]->dims[i] = in1->dims[i];
  }
  ele->add_int_property("inplace_a", inplace_a);
  layers.push_back(ele);
  return ele->outputs[0];
#ifdef DEADCODE
  ElementBinary *ele = new ElementBinary(*this, op, in1, in2, inplace_a, name);
  layers.push_back(ele);
  return ele->outputs[0];
#endif
}

Op* ElementBinary::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("inplace_a", value);
  bool inplace_a = (bool) value;
  return new ElementBinary(model, layer->op_type, inputs[0], inputs[1],
      inplace_a, layer->name);
}

Tensor FFModel::add(const Tensor in1,
                    const Tensor in2,
                    bool inplace_a,
                    char const *name)
{
  return this->binary(OP_EW_ADD, in1, in2, inplace_a, name);
}

Tensor FFModel::subtract(const Tensor in1,
                         const Tensor in2,
                         bool inplace_a,
                         char const *name)
{
  return this->binary(OP_EW_SUB, in1, in2, inplace_a, name);
}

Tensor FFModel::multiply(const Tensor in1,
                         const Tensor in2,
                         bool inplace_a,
                         char const *name)
{
  return this->binary(OP_EW_MUL, in1, in2, inplace_a, name);
}

Tensor FFModel::divide(const Tensor in1,
                       const Tensor in2,
                       bool inplace_a,
                       char const *name)
{
  return this->binary(OP_EW_DIV, in1, in2, inplace_a, name);
}

ElementBinary::ElementBinary(FFModel& model,
                             OperatorType _op_type,
                             const ParallelTensor in1,
                             const ParallelTensor in2,
                             bool _inplace_a,
                             const char* name)
: Op(
    model,
    _op_type,
    name,
    2/*inputs*/,
    0/*weights*/,
    1/*outputs*/,
    in1,
    in2
  ),
  inplace_a(_inplace_a)
{
  //TODO: implement broadcast op
  numOutputs = 1;
  numWeights = 0;
  assert(in1->num_dims == in2->num_dims);
  assert(in1->data_type == in2->data_type);
  int numdim = in1->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    assert(in1->dims[i] == in2->dims[i]);
    dims[i] = in1->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(numdim, dims, in1->data_type, this);
}

void ElementBinary::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(ElementBinary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  if (!inplace_a) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(2, FID_DATA);
  } else {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
  }
  //launcher.add_region_requirement(
  //  RegionRequirement(input_grad_lps[0], 0/*projection id*/,
  //    WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  //launcher.add_field(3, FID_DATA);
  //if (inputs[0]->region_grad != inputs[1]->region_grad) {
    // regions[4](I/O): input1_grad
  //  launcher.add_region_requirement(
  //    RegionRequirement(input_grad_lps[1], 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[1]->region_grad));
  //  launcher.add_field(4, FID_DATA);
  //}
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void ElementBinary::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_FWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  if (inplace_a) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[1]->region));
    launcher.add_field(1, FID_DATA);
  } else {
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[1]->region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(2, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void ElementBinary::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  if (inplace_a) {
    // regions[0](I/O): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1](I): input0
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(1, FID_DATA);
    if (inputs[0]->region == inputs[1]->region) {
      // regions[3](I): input1
      launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[1]->region));
      launcher.add_field(2, FID_DATA);
      // regions[4](I/O): input1_grad
      launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
      launcher.add_field(3, FID_DATA);
    }
  } else {
    // regions[0](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1](I): input0
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(1, FID_DATA);
    // regions[2](I/O): input0_grad
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(2, FID_DATA);
    if (inputs[0]->region == inputs[1]->region) {
      // regions[3](I): input1
      launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[1]->region));
      launcher.add_field(3, FID_DATA);
      // regions[4](I/O): input1_grad
      launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
      launcher.add_field(4, FID_DATA);
    }
  }
  runtime->execute_index_space(ctx, launcher);
}

size_t ElementBinary::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->inputs[1]->get_owner_independent_hash());
  hash_combine(hash, this->op_type);

  return hash;
}

using PCG::Node;
Node FFModel::get_or_create_element_binary_node(const ParallelTensor input1,
                                                const ParallelTensor input2,
                                                OperatorType op_type)
{
  size_t hash = input1->get_owner_independent_hash();
  hash = hash * 31 + input2->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(op_type);
  const auto& it = cached_element_binary_ops.find(hash);
  ElementBinary* eb = NULL;
  if (it != cached_element_binary_ops.end()) {
    eb = it->second;
  } else {
    eb = new ElementBinary(*this, op_type, input1, input2, false/*inplace*/, NULL);
    cached_element_binary_ops[hash] = eb;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = eb;
  return ret;
}

}; // namespace FlexFlow
