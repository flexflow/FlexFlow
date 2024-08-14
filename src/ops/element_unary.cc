#include "flexflow/ops/element_unary.h"
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
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::unary(OperatorType op,
                      const Tensor x,
                      bool inplace,
                      char const *name,
                      float scalar) {
  Layer *ele = nullptr;
  DataType dtype = x->data_type;
  // if (x->data_type < DT_FLOAT) {
  if (false) {
    dtype = DT_FLOAT;
    std::string str = (name == nullptr) ? "" : std::string(name);
    Tensor new_x = cast(x, dtype, (str + "input_pre_cast").c_str());
    ele = new Layer(this,
                    op,
                    dtype,
                    name,
                    1 /*inputs*/,
                    0 /*weights*/,
                    1 /*outputs*/,
                    new_x);
  } else {
    dtype = x->data_type;
    ele = new Layer(
        this, op, dtype, name, 1 /*inputs*/, 0 /*weights*/, 1 /*outputs*/, x);
  }
  int numdims = x->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++) {
    dims[i] = x->dims[i];
  }
  ele->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, dtype, ele, 0, true /*create_grad*/);
  ele->add_int_property("inplace", inplace);
  ele->add_float_property("scalar", scalar);
  layers.push_back(ele);
  return ele->outputs[0];
}

Op *ElementUnary::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("inplace", value);
  bool inplace = (bool)value;
  float scalar;
  layer->get_float_property("scalar", scalar);
  return new ElementUnary(model,
                          layer->layer_guid,
                          layer->op_type,
                          inputs[0],
                          inplace,
                          layer->name,
                          scalar);
}

ElementUnaryParams ElementUnary::get_params() const {
  ElementUnaryParams params;
  params.op_type = this->op_type;
  params.inplace = this->inplace;
  params.scalar = this->scalar;
  params.layer_guid = this->layer_guid;
  return params;
}

Tensor FFModel::exp(const Tensor x, char const *name) {
  return this->unary(OP_EXP, x, false /*inplace*/, name);
}

Tensor FFModel::scalar_multiply(const Tensor x,
                                float const scalar,
                                bool inplace,
                                char const *name) {
  return this->unary(OP_SCALAR_MULTIPLY, x, inplace, name, scalar);
}

Tensor FFModel::scalar_add(const Tensor x,
                           float const scalar,
                           bool inplace,
                           char const *name) {
  return this->unary(OP_SCALAR_ADD, x, inplace, name, scalar);
}

Tensor FFModel::scalar_sub(const Tensor x,
                           float const scalar,
                           bool inplace,
                           char const *name) {
  return this->unary(OP_SCALAR_SUB, x, inplace, name, scalar);
}

Tensor FFModel::scalar_truediv(const Tensor x,
                               float const scalar,
                               bool inplace,
                               char const *name) {
  return this->unary(OP_SCALAR_TRUE_DIV, x, inplace, name, scalar);
}

Tensor FFModel::relu(const Tensor x, bool inplace, char const *name) {
  return this->unary(OP_RELU, x, inplace, name);
}

Tensor FFModel::sigmoid(const Tensor x, char const *name) {
  return this->unary(OP_SIGMOID, x, false /*inplace*/, name);
}

Tensor FFModel::tanh(const Tensor x, char const *name) {
  return this->unary(OP_TANH, x, false /*inplace*/, name);
}

Tensor FFModel::identity(const Tensor x, char const *name) {
  return this->unary(OP_IDENTITY, x, false /*inplace*/, name);
}

Tensor FFModel::gelu(const Tensor x, char const *name) {
  return this->unary(OP_GELU, x, false /*inplace*/, name);
}

Tensor FFModel::elu(const Tensor x, bool inplace, char const *name) {
  // Currently assume inplace is false
  assert(!inplace);
  return this->unary(OP_ELU, x, inplace, name);
}

Tensor FFModel::rsqrt(const Tensor x, bool inplace, char const *name) {
  return this->unary(OP_RSQRT, x, inplace, name);
}

Tensor FFModel::pow(const Tensor x,
                    float const exponent,
                    bool inplace,
                    char const *name) {
  return this->unary(OP_POW, x, inplace, name, exponent);
}

Tensor FFModel::sin(const Tensor x, char const *name) {
  return this->unary(OP_SIN, x, false /*inplace*/, name);
}

Tensor FFModel::cos(const Tensor x, char const *name) {
  return this->unary(OP_COS, x, false /*inplace*/, name);
}

bool ElementUnaryParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

bool operator==(ElementUnaryParams const &lhs, ElementUnaryParams const &rhs) {
  return lhs.op_type == rhs.op_type && lhs.scalar == rhs.scalar &&
         lhs.inplace == rhs.inplace;
}

ElementUnary::ElementUnary(FFModel &model,
                           LayerID const &_layer_guid,
                           OperatorType _op_type,
                           const ParallelTensor x,
                           bool _inplace,
                           char const *name,
                           float _scalar)
    : Op(model,
         _op_type,
         x->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         x),
      inplace(_inplace), scalar(_scalar) {
  layer_guid = _layer_guid;
  numOutputs = 1;
  int numdim = x->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = x->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, inputs[0]->data_type, this);
  // Disable inplace if shape mismatch
  if (outputs[0]->get_shape() != inputs[0]->get_shape()) {
    inplace = false;
  }
}

ElementUnary::ElementUnary(FFModel &model,
                           ElementUnaryParams const &params,
                           const ParallelTensor input,
                           char const *name)
    : ElementUnary(model,
                   params.layer_guid,
                   params.op_type,
                   input,
                   params.inplace,
                   params.name,
                   params.scalar) {}

void ElementUnary::map_output_tensors(FFModel &ff) {
  if (has_inplace_output()) {
    assert(numOutputs == 1);
    assert(outputs[0]->get_volume() == inputs[0]->get_volume());
    outputs[0]->parallel_is = inputs[0]->parallel_is;
    outputs[0]->region = inputs[0]->region;
    outputs[0]->part = inputs[0]->part;
    outputs[0]->region_grad = inputs[0]->region_grad;
    outputs[0]->part_grad = inputs[0]->part_grad;
  } else {
    Op::map_output_tensors(ff);
  }
}

bool ElementUnary::can_inplace_output(void) {
  return outputs[0]->get_shape() == inputs[0]->get_shape();
}

bool ElementUnary::has_inplace_output(void) {
  return inplace;
}

void ElementUnary::do_inplace_output(void) {
  inplace = true;
}

bool ElementUnary::use_cudnn(OperatorType type) {
  if (type == OP_RELU) {
    return true;
  }
  if (type == OP_SIGMOID) {
    return true;
  }
  if (type == OP_TANH) {
    return true;
  }
  if (type == OP_ELU) {
    return true;
  }
  return false;
}

void ElementUnary::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(ELEMENTUNARY_INIT_TASK_ID,
                              parallel_is,
                              TaskArgument(this, sizeof(ElementUnary)),
                              argmap,
                              Predicate::TRUE_PRED,
                              false /*must*/,
                              0 /*mapper_id*/,
                              outputs[0]->machine_view.hash());
  if (!inplace) {
    init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                           0 /*projection id*/,
                                                           READ_ONLY,
                                                           EXCLUSIVE,
                                                           inputs[0]->region));
    init_launcher.add_field(0, FID_DATA);
    init_launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                           0 /*projection id*/,
                                                           WRITE_ONLY,
                                                           EXCLUSIVE,
                                                           outputs[0]->region));
    init_launcher.add_field(1, FID_DATA);
  } else {
    init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                           0 /*projection id*/,
                                                           READ_WRITE,
                                                           EXCLUSIVE,
                                                           inputs[0]->region));
    init_launcher.add_field(0, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void ElementUnary::init_inference(
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
  IndexLauncher init_launcher(ELEMENTUNARY_INIT_TASK_ID,
                              parallel_is,
                              TaskArgument(this, sizeof(ElementUnary)),
                              argmap,
                              Predicate::TRUE_PRED,
                              false /*must*/,
                              0 /*mapper_id*/,
                              machine_view_hash);
  if (!inplace) {
    init_launcher.add_region_requirement(
        RegionRequirement(batch_inputs[0]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          batch_inputs[0]->region));
    init_launcher.add_field(0, FID_DATA);
    init_launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    init_launcher.add_field(1, FID_DATA);
  } else {
    init_launcher.add_region_requirement(
        RegionRequirement(batch_inputs[0]->part,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          batch_inputs[0]->region));
    init_launcher.add_field(0, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

OpMeta *ElementUnary::init_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  ElementUnary *eu = (ElementUnary *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  ElementUnaryMeta *m = new ElementUnaryMeta(handle);
  m->op_type = eu->op_type;
  m->data_type = eu->outputs[0]->data_type;
  // Input and output should have the same data type
  assert(eu->outputs[0]->data_type == eu->inputs[0]->data_type);
  m->profiling = eu->profiling;
  m->inference_debugging = eu->inference_debugging;
  m->inplace = eu->inplace;
  m->scalar = eu->scalar;
  std::strcpy(m->op_name, eu->name);
  m->layer_guid = eu->layer_guid;
  if (m->inplace) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
  }

  if (use_cudnn(m->op_type)) {
    Domain input_domain = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());
    ElementUnary::init_kernel(m, input_domain, input_domain);
  }
  return m;
}

void ElementUnary::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(0, FID_DATA);
  } else {
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
  }
  runtime->execute_index_space(ctx, launcher);
}

FutureMap ElementUnary::inference(
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

  IndexLauncher launcher(ELEMENTUNARY_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  if (inplace) {
    assert(batch_outputs[0]->part == batch_inputs[0]->part);
    assert(batch_outputs[0]->region == batch_inputs[0]->region);
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(0, FID_DATA);
  } else {
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
  }
  return runtime->execute_index_space(ctx, launcher);
}

void ElementUnary::inference_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(task->regions.size() == regions.size());
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  ElementUnaryMeta const *m = *((ElementUnaryMeta **)task->local_args);
  if (m->data_type == DT_HALF) {
    forward_task_with_type<half>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

void ElementUnary::forward_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  ElementUnaryMeta const *m = *((ElementUnaryMeta **)task->local_args);
  if (m->data_type == DT_HALF) {
    forward_task_with_type<half>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
template <typename DT>
void ElementUnary::forward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  // const ElementUnary* ele = (const ElementUnary*) task->args;
  ElementUnaryMeta *m = *((ElementUnaryMeta **)task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  const DT *input_ptr = NULL;
  DT *output_ptr = NULL;
  if (m->inplace) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    output_ptr = helperGetTensorPointerRW<DT>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_ptr = output_ptr;
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain output_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    assert(output_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<DT>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerWO<DT>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
  }

  ElementUnary::forward_kernel_wrapper<DT>(
      m, input_ptr, output_ptr, input_domain.get_volume());

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    std::vector<GenericTensorAccessorR> input_accessors;
    std::vector<GenericTensorAccessorR> output_accessors;
    if (m->inplace) {
      GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
          m->data_type, regions[0], task->regions[0], FID_DATA, ctx, runtime);
      output_accessors.push_back(output);
    } else {
      GenericTensorAccessorR input = helperGetGenericTensorAccessorWO(
          m->data_type, regions[0], task->regions[0], FID_DATA, ctx, runtime);
      GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
          m->data_type, regions[1], task->regions[1], FID_DATA, ctx, runtime);
      input_accessors.push_back(input);
      output_accessors.push_back(output);
    }
    ElementUnary::save_inference_tensors_to_file(
        m, shard_id, nullptr, input_accessors, {}, output_accessors);
  }
}

void ElementUnary::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace) {
    assert(inputs[0]->part == outputs[0]->part);
    assert(inputs[0]->part_grad == outputs[0]->part_grad);
    // regions[2](I): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      outputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
  } else {
    // regions[0](I): input
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[1](I/O): input_grad
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
    // regions[2](I): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(2, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region_grad));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void ElementUnary::backward_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  ElementUnaryMeta const *m = *((ElementUnaryMeta **)task->local_args);
  if (m->data_type == DT_FLOAT) {
    backward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    backward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
template <typename DT>
void ElementUnary::backward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  // const ElementUnary* ele = (const ElementUnary*) task->args;
  ElementUnaryMeta const *m = *((ElementUnaryMeta **)task->local_args);
  const DT *input_ptr = NULL, *output_ptr = NULL, *output_grad_ptr = NULL;
  DT *input_grad_ptr = NULL;
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (m->inplace) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain input_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    assert(input_grad_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<DT>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<DT>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    output_ptr = input_ptr;
    output_grad_ptr = input_grad_ptr;
  } else {
    assert(regions.size() == 4);
    assert(task->regions.size() == 4);
    Domain input_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    Domain output_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    Domain output_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
    assert(output_grad_domain == input_domain);
    assert(output_grad_domain == output_domain);
    assert(output_grad_domain == input_grad_domain);
    input_ptr = helperGetTensorPointerRO<DT>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<DT>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerRO<DT>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    output_grad_ptr = helperGetTensorPointerRO<DT>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }

  ElementUnary::backward_kernel_wrapper<DT>(m,
                                            input_ptr,
                                            input_grad_ptr,
                                            output_ptr,
                                            output_grad_ptr,
                                            input_domain.get_volume());
}

void ElementUnary::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->op_type);
  sez.serialize(this->inplace);
  sez.serialize(scalar);
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

bool ElementUnary::measure_operator_cost(Simulator *sim,
                                         MachineView const &mv,
                                         CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  ElementUnaryMeta *m = sim->ele_unary_meta;
  m->op_type = op_type;
  if (use_cudnn(m->op_type)) {
    Domain input_domain, output_domain;
    input_domain.dim = sub_input.num_dims;
    for (int i = 0; i < sub_input.num_dims; i++) {
      input_domain.rect_data[i] = 0;
      input_domain.rect_data[i + input_domain.dim] = sub_input.dims[i].size - 1;
    }
    output_domain.dim = sub_output.num_dims;
    for (int i = 0; i < sub_output.num_dims; i++) {
      output_domain.rect_data[i] = 0;
      output_domain.rect_data[i + input_domain.dim] =
          sub_output.dims[i].size - 1;
    }
    init_kernel(m, input_domain, output_domain);
  }
  sim->free_all();
  float *input_ptr =
      (float *)sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = NULL;
  if (inplace) {
    output_ptr = input_ptr;
  } else {
    output_ptr =
        (float *)sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  }
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, input_ptr, output_ptr, sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr = NULL;
    if (inplace) {
      output_grad_ptr = input_grad_ptr;
    } else {
      output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(),
                                               outputs[0]->data_type);
    }
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [=] {
      backward_kernel_wrapper(m,
                              input_ptr,
                              input_grad_ptr,
                              output_ptr,
                              output_grad_ptr,
                              sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time);
  }
  return true;
}

using PCG::Node;
/*static*/
Node ElementUnary::deserialize(FFModel &ff,
                               Legion::Deserializer &dez,
                               ParallelTensor inputs[],
                               int num_inputs) {
  assert(num_inputs == 1);
  OperatorType op_type;
  float scalar;
  bool inplace;
  dez.deserialize(op_type);
  dez.deserialize(inplace);
  dez.deserialize(scalar);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  ElementUnaryParams params;
  params.op_type = op_type;
  params.inplace = inplace;
  params.scalar = scalar;
  params.layer_guid = layer_guid;
  strcpy(params.name, name);
  return ff.get_or_create_node<ElementUnary>(inputs[0], params);
}

Op *ElementUnary::materialize(FFModel &ff,
                              ParallelTensor inputs[],
                              int num_inputs) const {
  assert(num_inputs == 1);
  return new ElementUnary(ff,
                          this->layer_guid,
                          this->op_type,
                          inputs[0],
                          this->inplace,
                          this->name,
                          this->scalar);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ElementUnaryParams>::operator()(
    FlexFlow::ElementUnaryParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.op_type);
  hash_combine(key, params.scalar);
  hash_combine(key, params.inplace);
  hash_combine(key, params.layer_guid.id);
  return key;
}
}; // namespace std
