#include "flexflow/ops/element_binary.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/element_binary_kernels.h"
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

using namespace FlexFlow::Kernels::ElementBinary;

bool broadcastable(const Tensor t1, const Tensor t2) {
  int dim = std::min(t1->num_dims, t2->num_dims);
  for (int i = 0; i < dim; i++) {
    if ((t1->dims[i] != t2->dims[i]) && (t1->dims[i] > 1) &&
        (t2->dims[i] > 1)) {
      return false;
    }
  }
  return true;
}

Tensor FFModel::binary(OperatorType op,
                       const Tensor in1,
                       const Tensor in2,
                       bool inplace_a,
                       char const *name) {
  Layer *ele = nullptr;
  DataType dtype;
  assert(broadcastable(in1, in2));
  if (in1->data_type < in2->data_type) {
    dtype = in2->data_type;
    std::string str;
    if (name != nullptr) {
      str = std::string(name) + "input1_pre_cast";
    }
    Tensor new_in1 = cast(in1, dtype, str.c_str());
    ele = new Layer(this,
                    op,
                    dtype,
                    name,
                    2 /*inputs*/,
                    0 /*weights*/,
                    1 /*outputs*/,
                    new_in1,
                    in2);
  } else if (in1->data_type > in2->data_type) {
    dtype = in1->data_type;
    std::string str;
    if (name != nullptr) {
      str = std::string(name) + "input2_pre_cast";
    }
    Tensor new_in2 = cast(in2, dtype, str.c_str());
    ele = new Layer(this,
                    op,
                    dtype,
                    name,
                    2 /*inputs*/,
                    0 /*weights*/,
                    1 /*outputs*/,
                    in1,
                    new_in2);
  } else {
    dtype = in1->data_type;
    ele = new Layer(this,
                    op,
                    dtype,
                    name,
                    2 /*inputs*/,
                    0 /*weights*/,
                    1 /*outputs*/,
                    in1,
                    in2);
  }
  // Assert type match after broadcast
  assert(ele->inputs[0]->data_type == ele->inputs[1]->data_type);

  int numdim = in1->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    if (in1->dims[i] == 1) {
      dims[i] = in2->dims[i];
    } else if (in2->dims[i] == 1) {
      dims[i] = in1->dims[i];
    } else {
      dims[i] = in1->dims[i];
    }
  }

  ele->outputs[0] = create_tensor_legion_ordering(
      in1->num_dims, dims, ele->data_type, ele, 0, true /*create_grad*/);
  ele->add_int_property("inplace_a", inplace_a);
  layers.push_back(ele);
  return ele->outputs[0];
}

Op *ElementBinary::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("inplace_a", value);
  bool inplace_a = (bool)value;
  return new ElementBinary(model,
                           layer->layer_guid,
                           layer->op_type,
                           inputs[0],
                           inputs[1],
                           inplace_a,
                           layer->name);
}

Tensor FFModel::add(const Tensor in1,
                    const Tensor in2,
                    bool inplace_a,
                    char const *name) {
  return this->binary(OP_EW_ADD, in1, in2, inplace_a, name);
}

Tensor FFModel::subtract(const Tensor in1,
                         const Tensor in2,
                         bool inplace_a,
                         char const *name) {
  return this->binary(OP_EW_SUB, in1, in2, inplace_a, name);
}

Tensor FFModel::multiply(const Tensor in1,
                         const Tensor in2,
                         bool inplace_a,
                         char const *name) {
  return this->binary(OP_EW_MUL, in1, in2, inplace_a, name);
}

Tensor FFModel::divide(const Tensor in1,
                       const Tensor in2,
                       bool inplace_a,
                       char const *name) {
  return this->binary(OP_EW_DIV, in1, in2, inplace_a, name);
}

Tensor FFModel::max(const Tensor in1,
                    const Tensor in2,
                    bool inplace_a,
                    char const *name) {
  return this->binary(OP_EW_MAX, in1, in2, inplace_a, name);
}

Tensor FFModel::min(const Tensor in1,
                    const Tensor in2,
                    bool inplace_a,
                    char const *name) {
  return this->binary(OP_EW_MIN, in1, in2, inplace_a, name);
}

bool ElementBinaryParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  bool is_valid = true;
  is_valid &= (input.first.is_valid() & input.second.is_valid());
  if (!is_valid) {
    return false;
  }
  // is_valid &= (input.first == input.second);
  ParallelTensorShape A = input.first;
  ParallelTensorShape B = input.second;
  int numdim = std::min(A.num_dims, B.num_dims);
  for (int i = 0; i < numdim; i++) {
    if (A.dims[i].size > 1 && B.dims[i].size > 1) {
      if (A.dims[i] != B.dims[i]) {
        return false;
      }
    }
  }
  return is_valid;
}

bool operator==(ElementBinaryParams const &lhs,
                ElementBinaryParams const &rhs) {
  return lhs.type == rhs.type && lhs.layer_guid == rhs.layer_guid &&
         lhs.inplace_a == rhs.inplace_a;
}

ElementBinary::ElementBinary(FFModel &model,
                             LayerID const &_layer_guid,
                             OperatorType _op_type,
                             const ParallelTensor in1,
                             const ParallelTensor in2,
                             bool _inplace_a,
                             char const *name)
    : Op(model,
         _op_type,
         in1->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         in1,
         in2),
      inplace_a(_inplace_a) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  numOutputs = 1;
  numWeights = 0;
  assert(in1->data_type == in2->data_type);
  int numdim = std::max(in1->num_dims, in2->num_dims);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    if (i >= in1->num_dims) {
      dims[i] = in2->dims[i];
    } else if (i >= in2->num_dims) {
      dims[i] = in1->dims[i];
    } else if (in1->dims[i].size == in2->dims[i].size) {
      assert(in1->dims[i] == in2->dims[i]);
      dims[i] = in1->dims[i];
    } else if (in1->dims[i].size == 1) {
      dims[i] = in2->dims[i];
    } else if (in2->dims[i].size == 1) {
      dims[i] = in1->dims[i];
    } else {
      assert(false && "Operands could not be broadcast together");
      exit(0);
    }
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, in1->data_type, this);
  broadcast_input1 = (inputs[0]->get_volume() != outputs[0]->get_volume());
  broadcast_input2 = (inputs[1]->get_volume() != outputs[0]->get_volume());
}

ElementBinary::ElementBinary(
    FFModel &model,
    ElementBinaryParams const &params,
    std::pair<ParallelTensor, ParallelTensor> const &inputs,
    char const *name)
    : ElementBinary(model,
                    params.layer_guid,
                    params.type,
                    inputs.first,
                    inputs.second,
                    params.inplace_a,
                    params.name) {}

void ElementBinary::map_output_tensors(FFModel &ff) {
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

bool ElementBinary::can_inplace_output(void) {
  if (op_type == OP_EW_ADD || op_type == OP_EW_MUL) {
    // TODO: Currently assume that we always inplace_a
    if (outputs[0]->num_dims != inputs[0]->num_dims) {
      return false;
    }
    for (int i = 0; i < inputs[0]->num_dims; i++) {
      if (inputs[0]->dims[i] != outputs[0]->dims[i]) {
        return false;
      }
    }
    return outputs[0]->get_shape() == inputs[0]->get_shape();
  }
  return false;
}

bool ElementBinary::has_inplace_output(void) {
  return inplace_a;
}

void ElementBinary::do_inplace_output(void) {
  inplace_a = true;
}

void ElementBinary::init_inference(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  // Check if we have the same oprands
  has_same_operands = (batch_inputs[0]->region == batch_inputs[1]->region);
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(ELEMENTBINARY_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ElementBinary)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  int rid = 0;
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(rid++, FID_DATA);
  if (!has_same_operands) {
    launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      batch_inputs[1]->region));
    launcher.add_field(rid++, FID_DATA);
  } else {
    assert(batch_inputs[0]->part == batch_inputs[1]->part);
  }
  if (!inplace_a) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
  } else {
    assert(batch_outputs[0]->part == batch_inputs[0]->part);
    assert(batch_outputs[0]->region == batch_inputs[0]->region);
  }
  // launcher.add_region_requirement(
  //   RegionRequirement(input_grad_lps[0], 0/*projection id*/,
  //     WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  // launcher.add_field(3, FID_DATA);
  // if (inputs[0]->region_grad != inputs[1]->region_grad) {
  //  regions[4](I/O): input1_grad
  //  launcher.add_region_requirement(
  //    RegionRequirement(input_grad_lps[1], 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[1]->region_grad));
  //  launcher.add_field(4, FID_DATA);
  //}
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void ElementBinary::init(FFModel const &ff) {
  // Check if we have the same oprands
  has_same_operands = (inputs[0]->region == inputs[1]->region);
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ElementBinary)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  int rid = 0;
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(rid++, FID_DATA);
  if (!has_same_operands) {
    launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[1]->region));
    launcher.add_field(rid++, FID_DATA);
  } else {
    assert(inputs[0]->part == inputs[1]->part);
  }
  if (!inplace_a) {
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
  } else {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
  }
  // launcher.add_region_requirement(
  //   RegionRequirement(input_grad_lps[0], 0/*projection id*/,
  //     WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  // launcher.add_field(3, FID_DATA);
  // if (inputs[0]->region_grad != inputs[1]->region_grad) {
  //  regions[4](I/O): input1_grad
  //  launcher.add_region_requirement(
  //    RegionRequirement(input_grad_lps[1], 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[1]->region_grad));
  //  launcher.add_field(4, FID_DATA);
  //}
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *ElementBinary::init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  ElementBinary *eb = (ElementBinary *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  ElementBinaryMeta *m = new ElementBinaryMeta(handle, eb);
  for (int i = 0; i < eb->numInputs; i++) {
    m->trainable_inputs[i] = eb->trainable_inputs[i];
  }
  m->op_type = eb->op_type;
  m->profiling = eb->profiling;
  m->inference_debugging = eb->inference_debugging;
  m->inplace_a = eb->inplace_a;
  m->has_same_operands = eb->has_same_operands;
  m->broadcast_input1 = eb->broadcast_input1;
  m->broadcast_input2 = eb->broadcast_input2;
  std::strcpy(m->op_name, eb->name);
  m->layer_guid = eb->layer_guid;
  Domain input1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain input2_domain, output_domain;
  size_t num_regions = 1;
  if (!m->has_same_operands) {
    input2_domain = runtime->get_index_space_domain(
        ctx, task->regions[num_regions].region.get_index_space());
    num_regions++;
  } else {
    input2_domain = input1_domain;
  }
  if (!m->inplace_a) {
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[num_regions].region.get_index_space());
    num_regions++;
    // check that input can broadcast to output
    for (int i = 0; i < output_domain.dim; i++) {
      int output_dim_size = output_domain.hi()[i] - output_domain.lo()[i] + 1;
      if (i < input1_domain.dim) {
        int input1_dim_size = input1_domain.hi()[i] - input1_domain.lo()[i] + 1;
        assert(input1_dim_size == output_dim_size || input1_dim_size == 1);
      }
      if (i < input2_domain.dim) {
        int input2_dim_size = input2_domain.hi()[i] - input2_domain.lo()[i] + 1;
        assert(input2_dim_size == output_dim_size || input2_dim_size == 1);
      }
    }
  } else {
    output_domain = input1_domain;
  }
  assert(task->regions.size() == regions.size());
  assert(regions.size() == num_regions);
  init_kernel(m, input1_domain, input2_domain, output_domain);
  return m;
}

void ElementBinary::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace_a) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    if (has_same_operands) {
      // do nothing else
    } else {
      launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        inputs[1]->region));
      launcher.add_field(1, FID_DATA);
    }
  } else {
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    if (has_same_operands) {
      launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                        0 /*projection id*/,
                                                        WRITE_ONLY,
                                                        EXCLUSIVE,
                                                        outputs[0]->region));
      launcher.add_field(1, FID_DATA);
    } else {
      launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        inputs[1]->region));
      launcher.add_field(1, FID_DATA);
      launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                        0 /*projection id*/,
                                                        WRITE_ONLY,
                                                        EXCLUSIVE,
                                                        outputs[0]->region));
      launcher.add_field(2, FID_DATA);
    }
  }
  runtime->execute_index_space(ctx, launcher);
}

FutureMap
    ElementBinary::inference(FFModel const &ff,
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
  /* std::cout << "ElementBinary op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(ELEMENTBINARY_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  if (inplace_a) {
    assert(batch_outputs[0]->part == batch_inputs[0]->part);
    assert(batch_outputs[0]->region == batch_inputs[0]->region);
    launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      batch_inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    if (has_same_operands) {
      // do nothing else
    } else {
      launcher.add_region_requirement(
          RegionRequirement(batch_inputs[1]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            batch_inputs[1]->region));
      launcher.add_field(1, FID_DATA);
    }
  } else {
    launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      batch_inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    if (has_same_operands) {
      launcher.add_region_requirement(
          RegionRequirement(batch_outputs[0]->part,
                            0 /*projection id*/,
                            WRITE_ONLY,
                            EXCLUSIVE,
                            batch_outputs[0]->region));
      launcher.add_field(1, FID_DATA);
    } else {
      launcher.add_region_requirement(
          RegionRequirement(batch_inputs[1]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            batch_inputs[1]->region));
      launcher.add_field(1, FID_DATA);
      launcher.add_region_requirement(
          RegionRequirement(batch_outputs[0]->part,
                            0 /*projection id*/,
                            WRITE_ONLY,
                            EXCLUSIVE,
                            batch_outputs[0]->region));
      launcher.add_field(2, FID_DATA);
    }
  }
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): in1
  regions[1](I): in2
  regions[2](O): output
*/
__host__ void
    ElementBinary::inference_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(task->regions.size() == regions.size());
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  // const ElementBinary* ele = (const ElementBinary*) task->args;
  ElementBinaryMeta *m = *((ElementBinaryMeta **)task->local_args);
  GenericTensorAccessorR in1, in2;
  GenericTensorAccessorW out;
  Domain in1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  if (!m->has_same_operands) {
    Domain in2_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    // Currently only support broadcast for add and sub
    if (in1_domain != in2_domain) {
      assert(m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD ||
             m->op_type == OP_EW_MUL);
    }
  }

  if (m->inplace_a) {
    if (m->has_same_operands) {
      assert(regions.size() == 1);
      assert(task->regions.size() == 1);
      out = helperGetGenericTensorAccessorRW(m->output_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = out;
      in1 = out;
    } else {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      out = helperGetGenericTensorAccessorRW(m->output_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = helperGetGenericTensorAccessorRO(m->input_type[1],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in1 = out;
    }
  } else {
    if (m->has_same_operands) {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      in1 = helperGetGenericTensorAccessorRO(m->input_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = in1;
      out = helperGetGenericTensorAccessorWO(m->output_type[0],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
    } else {
      assert(regions.size() == 3);
      assert(task->regions.size() == 3);
      in1 = helperGetGenericTensorAccessorRO(m->input_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = helperGetGenericTensorAccessorRO(m->input_type[1],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      out = helperGetGenericTensorAccessorWO(m->output_type[0],
                                             regions[2],
                                             task->regions[2],
                                             FID_DATA,
                                             ctx,
                                             runtime);
    }
  }
  forward_kernel_wrapper(m, in1, in2, out);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    std::vector<GenericTensorAccessorR> weights_accessors;
    ElementBinary::save_inference_tensors_to_file(
        m, shard_id, bc, {in1, in2}, {}, {out});
  }
}

/*
  regions[0](I): in1
  regions[1](I): in2
  regions[2](O): output
*/
__host__ void
    ElementBinary::forward_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  // const ElementBinary* ele = (const ElementBinary*) task->args;
  ElementBinaryMeta const *m = *((ElementBinaryMeta **)task->local_args);
  GenericTensorAccessorR in1, in2;
  GenericTensorAccessorW out;
  Domain in1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  if (!m->has_same_operands) {
    Domain in2_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    // Currently only support broadcast for add and sub
    if (in1_domain != in2_domain) {
      assert(m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD ||
             m->op_type == OP_EW_MUL);
    }
  }

  if (m->inplace_a) {
    if (m->has_same_operands) {
      assert(regions.size() == 1);
      assert(task->regions.size() == 1);
      out = helperGetGenericTensorAccessorRW(m->output_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = out;
      in1 = out;
    } else {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      out = helperGetGenericTensorAccessorRW(m->output_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = helperGetGenericTensorAccessorRO(m->input_type[1],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in1 = out;
    }
  } else {
    if (m->has_same_operands) {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      in1 = helperGetGenericTensorAccessorRO(m->input_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = in1;
      out = helperGetGenericTensorAccessorWO(m->output_type[0],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
    } else {
      assert(regions.size() == 3);
      assert(task->regions.size() == 3);
      in1 = helperGetGenericTensorAccessorRO(m->input_type[0],
                                             regions[0],
                                             task->regions[0],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      in2 = helperGetGenericTensorAccessorRO(m->input_type[1],
                                             regions[1],
                                             task->regions[1],
                                             FID_DATA,
                                             ctx,
                                             runtime);
      out = helperGetGenericTensorAccessorWO(m->output_type[0],
                                             regions[2],
                                             task->regions[2],
                                             FID_DATA,
                                             ctx,
                                             runtime);
    }
  }

  forward_kernel_wrapper(m, in1, in2, out);
}

void ElementBinary::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ELEMENTBINARY_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace_a) {
    // regions[0](I/O): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      outputs[0]->region_grad));
    launcher.add_field(0, FID_DATA);
    // regions[1](I): input0
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(1, FID_DATA);
    if (inputs[0]->region != inputs[1]->region) {
      // regions[3](I): input1
      launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        inputs[1]->region));
      launcher.add_field(2, FID_DATA);
      // regions[4](I/O): input1_grad
      launcher.add_region_requirement(
          RegionRequirement(inputs[1]->part_grad,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            inputs[1]->region_grad));
      launcher.add_field(3, FID_DATA);
    }
  } else {
    int rid = 0;
    // regions[0](I): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    // regions[1](I): input0
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[2](I/O): input0_grad
    if (trainable_inputs[0]) {
      launcher.add_region_requirement(
          RegionRequirement(inputs[0]->part_grad,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            inputs[0]->region_grad));
      launcher.add_field(rid++, FID_DATA);
    }
    if (inputs[0]->region != inputs[1]->region) {
      // regions[3](I): input1
      launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        inputs[1]->region));
      launcher.add_field(rid++, FID_DATA);
      // regions[4](I/O): input1_grad
      if (trainable_inputs[1]) {
        launcher.add_region_requirement(
            RegionRequirement(inputs[1]->part_grad,
                              0 /*projection id*/,
                              READ_WRITE,
                              EXCLUSIVE,
                              inputs[1]->region_grad));
        launcher.add_field(rid++, FID_DATA);
      }
    }
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I or I/O): out_grad (I/O if inplace_a)
  regions[1](I): in0
  regions[2](I/O): in0_grad (Missing if in0_grad = out_grad)
  regions[3](I): in1 (Missing if in0 = in1)
  regions[4](I/O): in1_grad (Missing if in0=in1)
*/
void ElementBinary::backward_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  // const ElementBinary* ele = (const ElementBinary*) task->args;
  ElementBinaryMeta const *m = *((ElementBinaryMeta **)task->local_args);
  float const *in0_ptr = NULL, *in1_ptr = NULL, *out_grad_ptr = NULL;
  float *in0_grad_ptr = NULL, *in1_grad_ptr = NULL;
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (m->inplace_a) {
    in0_grad_ptr = helperGetTensorPointerRW<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    assert(regions.size() == 2 || regions.size() == 4);
    assert(task->regions.size() == regions.size());
    if (regions.size() == 2) {
      Domain in0_domain = runtime->get_index_space_domain(
          ctx, task->regions[1].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
          regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
      out_grad_ptr = in0_grad_ptr;
    } else {
      Domain in0_domain = runtime->get_index_space_domain(
          ctx, task->regions[1].region.get_index_space());
      Domain in1_domain = runtime->get_index_space_domain(
          ctx, task->regions[2].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      // assert(in1_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
          regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = helperGetTensorPointerRO<float>(
          regions[2], task->regions[2], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
          regions[3], task->regions[3], FID_DATA, ctx, runtime);
      out_grad_ptr = in0_grad_ptr;
    }
  } else {
    int rid = 0;
    out_grad_ptr = helperGetTensorPointerRO<float>(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    rid++;
    Domain in0_domain = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
    in0_ptr = helperGetTensorPointerRO<float>(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    rid++;
    if (m->trainable_inputs[0]) {
      Domain in0_grad_domain = runtime->get_index_space_domain(
          ctx, task->regions[rid].region.get_index_space());
      assert(in0_domain == in0_grad_domain);
      in0_grad_ptr = helperGetTensorPointerRW<float>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
      rid++;
    }
    if (m->has_same_operands) {
      // in0 == in1
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
    } else {
      Domain in1_domain = runtime->get_index_space_domain(
          ctx, task->regions[rid].region.get_index_space());
      in1_ptr = helperGetTensorPointerRO<float>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
      rid++;
      if (m->trainable_inputs[1]) {
        Domain in1_grad_domain = runtime->get_index_space_domain(
            ctx, task->regions[rid].region.get_index_space());
        // assert(out_grad_domain == in1_domain);
        assert(in1_domain == in1_grad_domain);
        in1_grad_ptr = helperGetTensorPointerRW<float>(
            regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
        rid++;
      }
    }
    assert(task->regions.size() == rid);
    assert(task->regions.size() == regions.size());
  }

  backward_kernel_wrapper(
      m, out_grad_ptr, in0_ptr, in1_ptr, in0_grad_ptr, in1_grad_ptr);
}

bool ElementBinary::measure_operator_cost(Simulator *sim,
                                          MachineView const &mv,
                                          CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input1, sub_input2;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input1)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_input2)) {
    return false;
  }
  ElementBinaryMeta *m = new ElementBinaryMeta(sim->handler, this);
  m->op_type = op_type;
  m->profiling = this->profiling;
  m->inference_debugging = this->inference_debugging;
  m->inplace_a = this->inplace_a;
  m->has_same_operands = this->has_same_operands;
  m->broadcast_input1 = this->broadcast_input1;
  m->broadcast_input2 = this->broadcast_input2;
  Domain input1_domain = sub_input1.get_domain();
  Domain input2_domain = sub_input2.get_domain();
  Domain output_domain = sub_output.get_domain();

  init_kernel(m, input1_domain, input2_domain, output_domain);

  sim->free_all();
  float *input1_ptr = (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  GenericTensorAccessorR input1_acc(
      inputs[0]->data_type, input1_domain, input1_ptr);
  float *input2_ptr = (float *)sim->allocate(sub_input2.get_volume(), DT_FLOAT);
  assert(input2_ptr != NULL);
  GenericTensorAccessorR input2_acc(
      inputs[1]->data_type, input2_domain, input2_ptr);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = NULL;
  if (inplace_a) {
    output_ptr = input1_ptr;
  } else {
    output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  }
  assert(output_ptr != NULL);
  GenericTensorAccessorW output_acc(
      outputs[0]->data_type, output_domain, output_ptr);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, input1_acc, input2_acc, output_acc);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input1_grad_ptr =
        (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
    assert(input1_grad_ptr != NULL);
    float *input2_grad_ptr =
        (float *)sim->allocate(sub_input2.get_volume(), DT_FLOAT);
    assert(input2_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr = NULL;
    if (inplace_a) {
      output_grad_ptr = input1_grad_ptr;
    } else {
      output_grad_ptr =
          (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    }
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [=] {
      backward_kernel_wrapper(m,
                              output_grad_ptr,
                              input1_ptr,
                              input2_ptr,
                              input1_grad_ptr,
                              input2_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Elewise Binary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Elewise Binary] name(%s) num_elements(%zu) "
                      "forward_time(%.4lf)\n",
                      name,
                      sub_output.get_volume(),
                      cost_metrics.forward_time);
  }

  delete m;
  return true;
}

void ElementBinary::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->op_type);
  sez.serialize(this->inplace_a);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node ElementBinary::deserialize(FFModel &ff,
                                Legion::Deserializer &dez,
                                ParallelTensor inputs[],
                                int num_inputs) {
  assert(num_inputs == 2);
  OperatorType op_type;
  size_t id, transformer_layer_id, deserialized_model_id;
  bool inplace_a;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(op_type);
  dez.deserialize(inplace_a);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  ElementBinaryParams params;
  params.layer_guid = layer_guid;
  params.type = op_type;
  params.inplace_a = inplace_a;
  strcpy(params.name, name);
  return ff.get_or_create_node<ElementBinary>({inputs[0], inputs[1]}, params);
}

ElementBinaryParams ElementBinary::get_params() const {
  ElementBinaryParams params;
  params.layer_guid = this->layer_guid;
  params.type = this->op_type;
  params.inplace_a = this->inplace_a;
  return params;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ElementBinaryParams>::operator()(
    FlexFlow::ElementBinaryParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.type);
  hash_combine(key, params.inplace_a);
  return key;
}
}; // namespace std
