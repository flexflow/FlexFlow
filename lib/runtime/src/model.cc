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

#include "model.h"
/* #if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA) */
/* #include "flexflow/utils/cuda_helper.h" */
/* #else */
/* #include "utils/hip_helper.h" */
/* #endif */
#include "legion_parallel_tensor_shape.h"
#include "op-attrs/ffconst_utils.h"
#include "mapper.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "ops/aggregate.h"
#include "ops/aggregate_spec.h"
#include "ops/attention.h"
#include "ops/batch_matmul.h"
#include "ops/batch_norm.h"
#include "ops/cast.h"
#include "ops/concat.h"
#include "ops/conv_2d.h"
#include "ops/dropout.h"
#include "ops/element_binary.h"
#include "ops/element_unary.h"
#include "ops/embedding.h"
#include "ops/flat.h"
#include "ops/fused.h"
#include "ops/gather.h"
#include "ops/groupby.h"
#include "ops/layer_norm.h"
#include "ops/linear.h"
#include "ops/noop.h"
#include "ops/pool_2d.h"
#include "ops/reduce.h"
#include "ops/reshape.h"
#include "ops/reverse.h"
#include "ops/softmax.h"
#include "ops/split.h"
#include "ops/topk.h"
#include "ops/transpose.h"
#include "parallel_ops/combine.h"
#include "parallel_ops/fused_parallel_op.h"
#include "parallel_ops/partition.h"
#include "parallel_ops/reduction.h"
#include "parallel_ops/replicate.h"
#include "utils/random_utils.h"
#include "test_utils.h"
#include "legion/legion_utilities.h"
#include <dirent.h>
#include <queue>
#include <unordered_set>
#include "op_node.h"
#include "utils/containers.h"
#include "parallel_tensor_mapping.h"
#include "make_operator.h"
#include "op-attrs/ops/noop.h"

using namespace Legion;

namespace FlexFlow {

LegionRuntime::Logger::Category log_model("Model");
LegionRuntime::Logger::Category log_measure("measure");
LegionRuntime::Logger::Category log_profile("profile");


/* std::unordered_map<int, int> output_to_input_mapping( */
/*     std::vector<ParallelDimMappingRecord> const &mapping) { */
/*   std::unordered_map<int, int> dim_mapping; */
/*   for (ParallelDimMappingRecord const &record : mapping) { */
/*     if (record.get_type() == MappingRecordType::INPUT_OUTPUT) { */
/*       dim_mapping[record.output_dim] = record.input_dim; */
/*     } */
/*   } */

/*   return dim_mapping; */
/* } */

/* std::unordered_map<int, int> input_to_output_mapping( */
/*     std::vector<ParallelDimMappingRecord> const &mapping) { */
/*   std::unordered_map<int, int> dim_mapping; */
/*   for (ParallelDimMappingRecord const &record : mapping) { */
/*     if (record.get_type() == MappingRecordType::INPUT_OUTPUT) { */
/*       dim_mapping[record.input_dim] = record.output_dim; */
/*     } */
/*   } */

/*   return dim_mapping; */
/* } */

FFModel::FFModel(FFConfig const &_config, ComputationGraph const &cg, ParallelComputationGraph const &pcg)
    : op_global_guid(OP_GUID_FIRST_VALID),
      config(_config), 
      index_space_mgr(_config.legion_config), 
      computation_graph(cg),
      pcg(pcg) {

  Runtime *runtime = config.legion_config.lg_hlr;
  Context ctx = config.legion_config.lg_ctx;
  // Register machine views
  register_all_machine_views(config.numNodes,
                             config.workersPerNode,
                             config.cpusPerNode,
                             all_valid_views);
  metrics_input = -1;
  // Create field space
  {
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, config.legion_config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  ArgumentMap argmap;
  Rect<1> task_rect(Point<1>(0),
                    Point<1>(config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);

  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    FFInitInfo info;
    info.workSpaceSize = config.workSpaceSize;
    info.allowTensorOpMathConversion = config.allow_tensor_op_math_conversion;
    argmap.set_point(*it, TaskArgument(&info, sizeof(FFInitInfo)));
  }

  // Init CUDA library on each worker
  IndexLauncher initLauncher(FF_INIT_TASK_ID,
                             task_is,
                             TaskArgument(NULL, 0),
                             argmap,
                             Predicate::TRUE_PRED,
                             false /*must*/,
                             0 /*mapper_id*/,
                             FFConfig::DataParallelism_GPU);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<FFHandler>(*it);
  }
}

#ifdef FF_USE_NCCL
ncclComm_t *FFModel::find_nccl_comms(MachineView const &view) const {
  size_t key = get_std_hash(view);
  if (contains_key(this->view_hash_to_nccl_comms, key)) {
    return this->view_hash_to_nccl_comms.at(key);
  } else {
    assert (config.computationMode == COMP_MODE_INFERENCE);
    return nullptr;
  }
}
#endif

// Tensor FFModel::create_constant(TensorShape const &shape,
//                                 float value) {
//   // FIXME: currently create gradients for constants since the current auto grad
//   // algorithm computes gradients for all operators
//   Tensor tensor = create_tensor(shape, false /*create_grad*/);
//   tensor->initializer = new ConstantInitializer(value);
//   return tensor;
// }

OpNode FFModel::new_node(Op const *op) {
  return this->op_node_mgr.create(op);
}

// Tensor FFModel::create_tensor(TensorShape const &shape,
//                               bool create_grad) {
//   switch (shape.num_dims()) {
// #define DIMFUNC(DIM)                                                           \
//   case DIM:                                                                    \
//     return create_tensor<DIM>(shape, create_grad);
//     LEGION_FOREACH_N(DIMFUNC)
// #undef DIMFUNC
//     default:
//       assert(false && "Unsupported dim!");
//   }
// }

/* ParallelTensor FFModel::create_parallel_tensor(ParallelTensorShape const &shape, */ /*                                                bool create_grad, */
/*                                                size_t input_tensor_guid) { */
/*   switch (shape.num_dims()) { */
/* #define DIMFUNC(DIM)                                                           \ */
/*   case DIM:                                                                    \ */
/*     return create_parallel_tensor<DIM>(                                        \ */
/*         shape, create_grad, input_tensor_guid); */
/*     LEGION_FOREACH_N(DIMFUNC) */
/* #undef DIMFUNC */
/*     default: */
/*       assert(false && "Unsupported dim!"); */
/*   } */
/* } */

/* Tensor FFModel::create_tensor(LegionTensorShape const &shape, */
/*                                               bool create_grad) { */
/*   return create_tensor(static_cast<TensorShape>(shape), create_grad); */
/* } */

/* ParallelTensor */
/*     FFModel::create_parallel_tensor(LegionParallelTensorShape const &shape, */
/*                                                     bool create_grad, */
/*                                                     size_t input_tensor_guid) { */
/*   return this->create_parallel_tensor(static_cast<ParallelTensorShape>(shape), create_grad, input_tensor_guid); */
/* } */

/* template <int NDIM> */
/* Tensor FFModel::create_tensor(TensorShape const &shape, */
/*                               bool create_grad) { */
/*   assert (shape.num_dims() == NDIM); */
/*   Tensor tensor = this->tensor_mgr.create(shape, create_grad); */
/*   Layer *input_layer = this->layer_mgr.create(InputAttrs{}, shape.data_type, "input", {}, {}, {tensor}); */
/*   this->tensor_uses.update(*input_layer); */

/*   return tensor; */
/* } */

/* template <int NDIM> */
/* ParallelTensor FFModel::create_parallel_tensor(ParallelTensorShape const &shape, */
/*                                                bool create_grad, */
/*                                                size_t input_tensor_guid) { */
/*   assert (shape.num_dims() == NDIM); */
/*   ParallelTensor tensor = this->parallel_tensor_mgr.create(shape, create_grad); */

/*   if (owner_op == nullptr) { */
/*     NoOp *input_op = new NoOp(*this, OP_INPUT, input_tensor_guid, tensor); */
/*     operators.push_back(input_op); */
/*     tensor->owner_op = input_op; */
/*     tensor->owner_idx = 0; */
/*   } else { */
/*     tensor->owner_op = owner_op; */
/*     tensor->owner_idx = owner_idx; */
/*   } */

/*   assert(tensor->check_valid()); */
/*   return tensor; */
/* } */

/* Parameter FFModel::create_weight(LegionTensorShape const &shape, */
/*                                                  Layer const *layer, */
/*                                                  bool create_grad, */
/*                                                  Initializer *initializer, */
/*                                                  ParameterSyncType sync_type) { */
/*   return this->create_weight(static_cast<TensorShape>(shape), layer, create_grad, initializer, sync_type); */
/* } */

//Parameter FFModel::create_weight(TensorShape const &shape,
//                                 Layer const *owner_layer,
//                                 bool create_grad,
//                                 Initializer *initializer,
//                                 ParameterSyncType sync_type) {
//  assert(owner_layer != nullptr);
//  if (owner_layer == nullptr) {
//    owner_layer = this->layer_mgr.create(OP_WEIGHT, shape.data_type, nullptr, 0/*inputs*/, 0/*weights*/, 1/*outputs*/);
//  }
//
//  Parameter p = this->tensor_mgr.create(shape, create_grad, initializer, sync_type, owner_layer, 0/*owner_idx*/);
//
//  assert(p->get_volume() > 0);
//  return p;
//}

/* template <int NDIM> */
/* ParallelParameter FFModel::create_parallel_weight(ParallelTensorShape const &shape, */ 
/*                                                   Op const *owner_op, */
/*                                                   bool create_grad, */
/*                                                   Initializer *initializer, */
/*                                                   ParameterSyncType sync_type) { */
/*   ParallelParameter p = this->parallel_tensor_mgr.create( */
/*     shape, */
/*     create_grad, */
/*     sync_type, */
/*     initializer */
/*   ); */

/*   if (owner_op == NULL) { */
/*     NoOp *weight_op = new NoOp(*this, OP_WEIGHT, p); */
/*     operators.push_back(weight_op); */
/*     p->owner_op = weight_op; */
/*   } else { */
/*     p->owner_op = owner_op; */
/*   } */
/*   p->owner_idx = 0; */

/*   assert(p->get_volume() > 0); */
/*   assert(p->check_valid()); */
/*   return p; */
/* } */

/* ParallelParameter FFModel::create_parallel_weight(ParallelTensorShape const &shape, */
/*                                                   Op const *owner_op, */
/*                                                   bool create_grad, */
/*                                                   Initializer *initializer, */
/*                                                   ParameterSyncType sync_type) { */
/*   switch (shape.num_dims()) { */
/* #define DIMFUNC(DIM)                                                           \ */
/*   case DIM:                                                                    \ */
/*     return create_parallel_weight<DIM>(                                        \ */
/*         shape, owner_op, create_grad, initializer, sync_type); */
/*     LEGION_FOREACH_N(DIMFUNC) */
/* #undef DIMFUNC */
/*     default: */
/*       assert(false && "Unsupported dim!"); */
/*   } */
/* } */

/* ParallelParameter FFModel::create_parallel_weight( */
/*     LegionParallelTensorShape const &shape, */
/*     Op const *owner_op, */
/*     bool create_grad, */
/*     Initializer *initializer, */
/*     ParameterSyncType sync_type) { */

/*   return this->create_parallel_weight(static_cast<ParallelTensorShape>(shape), owner_op, create_grad, initializer, sync_type); */
/* } */


optional<ParallelTensor> FFModel::get_parallel_tensor_from_tensor(
    Tensor const &tensor) const {
  std::vector<parallel_tensor_guid_t> pt_guids = this->tensor_map.at(tensor);
  if (pt_guids.size() == 0) {
    return nullopt;
  } else {
    return this->parallel_tensor_mgr.at(get_only(pt_guids));
  }
  // check if tensor->parallel_tensor is already set
  // if (tensor->parallel_tensor != nullopt) {
  //   return tensor->parallel_tensor;
  // }
  // optional<TensorSourceInfo> source = this->model_spec.get_source(tensor);
  // if (source.has_value()) {
  //   Op const *mapped_op = nullptr;
  //   if (source->layer.op_type == OP_INPUT) {
  //     // We use tensor_guid to match input operators
  //     tensor_guid_t tensor_guid = this->model_spec.get_output(source->layer, 0)->guid;
  //     for (auto const &op : operators) {
  //       if (op->op_type == OP_INPUT) {
  //         if (tensor_guid == ((NoOp *)op)->input_tensor_guid) {
  //           assert(mapped_op == nullptr);
  //           mapped_op = op;
  //         }
  //       }
  //     }
  //   } else {
  //     for (auto const &op : operators) {
  //       if (op->layer_guid == source->layer.layer_guid) {
  //         assert(mapped_op == nullptr);
  //         mapped_op = op;
  //       }
  //     }
  //   }
  //   if (mapped_op != nullptr) {
  //     return mapped_op->outputs[source->idx];
  //   }
  // }
  // return nullopt;
}

void FFModel::reset_metrics() {
  Context ctx = config.legion_config.lg_ctx;
  Runtime *runtime = config.legion_config.lg_hlr;
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID,
                        TaskArgument(&metrics_op, sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);
}

void FFModel::init_operators() {
  for (auto const &op : operators) {
    op->init(*this);
  }
}

void FFModel::forward(int seq_length) {
  iter_config.seq_length = seq_length;
  for (auto const &op : operators) {
    op->forward(*this);
  }
}

void FFModel::recompile_on_condition(RecompileState &r) {
  if (r.trigger()) {
    r.alter();
  }
}

void FFModel::compute_metrics() {
  Operator final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  metrics_op->compute(this, final_operator->outputs[0], parallel_label_tensor.value());
}

void FFModel::get_metrics() {
  metrics_input = operators.size() - 1;
}

void FFModel::backward(int seq_length) {
  iter_config.seq_length = seq_length;
  assert(config.computationMode == COMP_MODE_TRAINING);
  // Compute metrics
  compute_metrics();
  // Compute the gradients of the final operator wrt loss
  Op const *final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  loss_op->backward(this, final_operator->outputs[0], parallel_label_tensor.value());
  // Perform backpropagation
  // std::set<LogicalRegion> resetedInputGrads;
  for (int l = operators.size() - 1; l >= 0; l--) {
#ifdef ENABLE_RESNET_INPUT_GRADIENT_OPTIMIZATION
    for (int i = 0; i < operators[l]->numInputs; i++) {
      if (resetedInputGrads.find(operators[l]->inputs[i]->region) ==
          resetedInputGrads.end()) {
        resetedInputGrads.insert(operators[l]->inputs[i]->region);
      } else {
        // This input's gradients has been reseted by other operators
        // So we should not do it again
        operators[l]->resetInputGrads[i] = false;
      }
    }
#endif
    // TODO: If operator serves for metrics and for further prop
    // if(l == metrics_input && metrics_input < (int)operators.size()-1)
    //  continue;
    operators[l]->backward(*this);
  }
}

void FFModel::update() {
  optimizer->next();
  for (size_t i = 0; i < parameters.size(); i++) {
    optimizer->update(parameters[i]);
  }
}

Op const *FFModel::get_final_operator() const {
  int idx = operators.size() - 1;
  std::vector<Op const *> operators = this->get_operators();
  while (operators[idx]->op_type == OP_INPUT ||
         operators[idx]->op_type == OP_WEIGHT) {
    idx--;
  }
  // assert that the final operator has exactly one output
  assert(operators[idx]->numOutputs == 1);
  return operators.at(idx);
}

void FFModel::compile(Optimizer *_optimizer,
                      LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  optimizer = _optimizer;
  compile(loss_type, metrics, comp_mode);
}

bool FFModel::apply_fusion(std::vector<Op *> const &operators,
                           std::vector<Op *> &new_operators) {
  // Context ctx = config.lg_ctx;
  // Runtime* runtime = config.lg_hlr;
  for (size_t l = 1; l < operators.size() - 1; l++) {
    // don't fuse input and weight operator since they don't involve any
    // forward/backward task launches
    if (operators[l]->op_type == OP_INPUT ||
        operators[l]->op_type == OP_WEIGHT) {
      continue;
    }
    // don't fuse parallel op since they have different parallel_is in
    // forward/backward
    if (operators[l]->is_parallel_op()) {
      continue;
    }
    size_t start = 0;
    {
      Op *opl = operators[l];
      for (int idx = 0; idx < opl->numInputs; idx++) {
        bool found = false;
        for (size_t i = 0; i < l; i++) {
          if (opl->inputs[idx]->owner_op == operators[i]) {
            assert(!found);
            found = true;
            if (i > start) {
              start = i;
            }
          }
        }
        assert(found || (opl->inputs[idx]->owner_op == NULL));
      }
    }
    for (size_t i = start; i < l; i++) {
      // Domain d1 =
      // runtime->get_index_space_domain(operators[l]->outputs[0]->parallel_is);
      // Domain d2 =
      // runtime->get_index_space_domain(operators[i]->outputs[0]->parallel_is);
      MachineView view1 = operators[l]->outputs[0]->machine_view.value();
      MachineView view2 = operators[i]->outputs[0]->machine_view.value();
      if (view1 == view2) {
        FusedOp *fused_op = nullptr;
        bool allocate_new_fused_op = false;
        if (operators[i]->op_type == OP_FUSED) {
          fused_op = (FusedOp *)operators[i];
        } else {
          //  cannot be an in-place operator
          if (operators[i]->has_inplace_output()) {
            continue;
          }
          // don't fuse input and weight operator since they don't involve any
          // forward/backward kernels
          if (operators[i]->op_type == OP_INPUT ||
              operators[i]->op_type == OP_WEIGHT) {
            continue;
          }
          // don't fuse parallel op since they have different parallel_is in
          // forward/backward
          if (operators[i]->is_parallel_op()) {
            continue;
          }
          fused_op = new FusedOp(*this, operators[i]);
          allocate_new_fused_op = true;
        }
        if (fused_op->add_operator(*this, operators[l])) {
          // Construct new operators
          new_operators.clear();
          for (size_t j = 0; j < i; j++) {
            new_operators.push_back(operators[j]);
          }
          new_operators.push_back(fused_op);
          for (size_t j = i + 1; j < operators.size(); j++) {
            if (j == l) {
              continue; // l and i are fused
            }
            Op *op = operators[j];
            // Update input tensors that belong to operator[l] or operator[i]
            for (int idx = 0; idx < op->numInputs; idx++) {
              if ((op->inputs[idx]->owner_op == operators[l]) ||
                  (op->inputs[idx]->owner_op == operators[i])) {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++) {
                  if (fused_op->outputs[k]->region == op->inputs[idx]->region) {
                    assert(found == -1);
                    found = k;
                  }
                }
                assert(found >= 0);
                op->inputs[idx] = fused_op->outputs[found];
              }
            }
            // Insert op
            new_operators.push_back(op);
          }
          // We are exact one operator fewer than the original
          assert(new_operators.size() + 1 == operators.size());
          return true;
        } else {
          // TODO: delete fused_op to avoid memory leakage
          if (allocate_new_fused_op) {
            delete fused_op;
          }
          continue;
        }
      }
    }
  }
  return false;
}

static ParallelTensorShape get_parallel_tensor_shape(Tensor const &tensor) {
  int num_dims = tensor->num_dims();
  std::vector<ParallelDim> dims;
  for (int j = 0; j < num_dims; j++) {
    dims.emplace_back(tensor->dims[j], 1, -1, false);
  }
  dims.emplace_back(1, 1, -1, true);
  ParallelTensorShape shape = { dims, tensor->data_type };
  return shape;
}

Op *FFModel::create_operator_from_layer(
    Layer *layer, std::vector<ParallelTensor> const &inputs) {
  return make_operator_unsafe(*this, layer->attrs, inputs);

  //switch (layer->op_type) {
  //  case OP_INPUT: {
  //    // Input op cannot have an input
  //    assert(inputs.size() == 0);
  //    Tensor tensor = layer->outputs[0];
  //    // Current assume we add one dimension before each tensor
  //    // create_parallel_tensor adds an NoOp into operators
  //    ParallelTensorShape shape = get_parallel_tensor_shape(tensor);
  //    ParallelTensor pt =
  //        create_parallel_tensor(shape,
  //                                               nullptr,
  //                                               0,
  //                                               true /*gradients*/,
  //                                               tensor->tensor_guid);
  //    // assert that this tensor hasn't been mapped before
  //    assert(tensor->parallel_tensor == nullopt);
  //    tensor->parallel_tensor = pt;
  //    // start from data parllel tensor
  //    if (config.only_data_parallel) {
  //      Repartition *part = new Repartition(
  //          *this, pt, shape.num_dims() - 1, config.numNodes * config.workersPerNode);
  //      operators.push_back(part);
  //    }
  //    return operators[operators.size() - 1];
  //  }
  //  case OP_MULTIHEAD_ATTENTION: {
  //    Op *op =
  //        MultiHeadAttention::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_BATCHMATMUL: {
  //    Op *op = BatchMatmul::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CAST: {
  //    Op *op = Cast::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CONCAT: {
  //    Op *op = Concat::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_CONV2D: {
  //    Op *op = Conv2D::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_DROPOUT: {
  //    Op *op = Dropout::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EMBEDDING: {
  //    Op *op = Embedding::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EW_ADD:
  //  case OP_EW_SUB:
  //  case OP_EW_MUL:
  //  case OP_EW_DIV:
  //  case OP_EW_MAX:
  //  case OP_EW_MIN: {
  //    Op *op = ElementBinary::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_EXP:
  //  case OP_SIN:
  //  case OP_COS:
  //  case OP_SCALAR_MULTIPLY:
  //  case OP_SCALAR_ADD:
  //  case OP_SCALAR_SUB:
  //  case OP_SCALAR_TRUE_DIV:
  //  case OP_POW:
  //  case OP_RELU:
  //  case OP_SIGMOID:
  //  case OP_TANH:
  //  case OP_IDENTITY:
  //  case OP_GELU:
  //  case OP_ELU: {
  //    Op *op = ElementUnary::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_FLAT: {
  //    Op *op = Flat::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_GATHER: {
  //    Op *op = Gather::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_LAYERNORM: {
  //    Op *op = LayerNorm::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_LINEAR: {
  //    Op *op = Linear::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_POOL2D: {
  //    Op *op = Pool2D::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_REDUCE_SUM: {
  //    Op *op = Reduce::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_RESHAPE: {
  //    Op *op = Reshape::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_SOFTMAX: {
  //    Op *op = Softmax::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_SPLIT: {
  //    Op *op = Split::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_TRANSPOSE: {
  //    Op *op = Transpose::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_TOPK: {
  //    Op *op = TopK::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_GROUP_BY: {
  //    Op *op = Group_by::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_AGGREGATE: {
  //    Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  case OP_AGG_SPEC: {
  //    Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
  //    operators.push_back(op);
  //    return op;
  //  }
  //  default:
  //    assert(false);
  //}
}

void FFModel::create_operators_from_layers() {
  std::map<Tensor const, ParallelTensor> tensors_to_parallel_tensors;
  for (auto const &l : layers) {
    std::vector<ParallelTensor> inputs;
    for (int i = 0; i < l->numInputs; i++) {
      // create new input tensors
      assert(tensors_to_parallel_tensors.find(l->inputs[i]) !=
             tensors_to_parallel_tensors.end());
      inputs.push_back(tensors_to_parallel_tensors[l->inputs[i]]);
    }
    Op *op = create_operator_from_layer(l, inputs);
    assert(op->numOutputs == l->numOutputs);
    for (int i = 0; i < op->numOutputs; i++) {
      tensors_to_parallel_tensors[l->outputs[i]] = op->outputs[i];
    }
  }
}

void FFModel::perform_inplace_optimizations() {
  for (size_t l = 1; l < operators.size(); l++) {
    if (operators[l]->can_inplace_output()) {
      // Assume outputs[0] is inplace with inputs[0]
      assert(operators[l]->numOutputs == 1);
      if (operators[l]->inputs[0]->owner_op != NULL) {
        // int dim1 = operators[l]->outputs[0]->num_dims;
        // int dim2 = operators[l]->inputs[0]->num_dims;
        MachineView view1 = operators[l]->outputs[0]->machine_view.value();
        MachineView view2 = operators[l]->inputs[0]->machine_view.value();
        if (view1 == view2) {
          // Check no others also need operators[l]->inputs[0]
          bool found = false;
          for (size_t i = 0; i < operators.size(); i++) {
            if (i == l) {
              continue;
            }
            for (int j = 0; j < operators[i]->numInputs; j++) {
              if ((operators[i]->inputs[j]->owner_op ==
                   operators[l]->inputs[0]->owner_op) &&
                  (operators[i]->inputs[j]->owner_idx ==
                   operators[l]->inputs[0]->owner_idx)) {
                found = true;
              }
            }
          }
          if (!found) {
            // Perform inplace
            operators[l]->do_inplace_output();
          }
        }
      }
    }
  }
}

void FFModel::perform_fusion_optimizations() {
  fprintf(stderr, "Applying fusion optimizations during compilation...\n");
  fprintf(stderr, "%zu operators before fusion...\n", operators.size());
  std::vector<Op *> new_operators;
  std::vector<Op *> old_operators = operators;
  while (apply_fusion(operators, new_operators)) {
    for (size_t i = 0; i < new_operators.size(); i++) {
      for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
        for (size_t j = i + 1; j < new_operators.size(); j++) {
          if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
            assert(false);
          }
        }
      }
    }
    operators = new_operators;
  }
  // Check integrity
  for (size_t l = 0; l < operators.size(); l++) {
    if (operators[l]->op_type == OP_FUSED) {
      FusedOp *fused = (FusedOp *)operators[l];
      int ioff = 0, woff = 0, ooff = 0;
      for (int op = 0; op < fused->numOperators; op++) {
        Op *old_op = fused->operators[op];
        for (int i = 0; i < fused->op_num_inputs[op]; i++) {
          int my_off = fused->op_input_idx[i + ioff];
          if (fused->op_input_source[i + ioff] == FusedOp::SOURCE_INPUT) {
            assert(fused->inputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else if (fused->op_input_source[i + ioff] ==
                     FusedOp::SOURCE_OUTPUT) {
            assert(fused->outputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else {
            assert(false);
          }
        }
        for (int i = 0; i < fused->op_num_weights[op]; i++) {
          int my_off = fused->op_weight_idx[i + woff];
          assert(fused->op_weight_source[i + woff] == FusedOp::SOURCE_WEIGHT);
          assert(fused->weights[my_off]->region ==
                 old_op->weights[i]->region);
        }
        for (int i = 0; i < fused->op_num_outputs[op]; i++) {
          int my_off = fused->op_output_idx[i + ooff];
          assert(fused->op_output_source[i + ooff] == FusedOp::SOURCE_OUTPUT);
          assert(fused->outputs[my_off]->region ==
                 old_op->outputs[i]->region);
        }
        ioff += fused->op_num_inputs[op];
        woff += fused->op_num_weights[op];
        ooff += fused->op_num_outputs[op];
      }
    } else {
      bool found = false;
      for (size_t i = 0; i < old_operators.size(); i++) {
        if (old_operators[i] == operators[l]) {
          assert(!found);
          found = true;
        }
      }
      assert(found);
    }
  }
  fprintf(stderr, "%zu operators after fusion...\n", operators.size());
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%s) guid(%lu)\n",
           i,
           get_operator_type_name(operators[i]->op_type).c_str(),
           operators[i]->op_guid);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numWeights; j++) {
      LogicalRegion handle = op->weights[j]->region;
      printf("weights[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}

void FFModel::initialize_nccl_communicators() {
  // init all nccl communicators
  Context ctx = this->config.legion_config.lg_ctx;
  Runtime *runtime = this->config.legion_config.lg_hlr;
  for (size_t l = 0; l < operators.size(); l++) {
    // Only create nccl for weights
    if (operators[l]->op_type != OP_WEIGHT) {
      continue;
    }
    MachineView view = operators[l]->outputs[0]->machine_view.value();
    if (view_hash_to_nccl_comms.find(get_std_hash(view)) ==
        view_hash_to_nccl_comms.end()) {
      TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
      Future future = runtime->execute_task(ctx, launcher);
      ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
      IndexSpace task_is = this->index_space_mgr.get_or_create_task_is(view);
      ArgumentMap argmap;
      IndexLauncher index_launcher(
          NCCL_INIT_COMMS_TASK_ID,
          task_is,
          TaskArgument(&ncclId, sizeof(ncclUniqueId)),
          argmap,
          Predicate::TRUE_PRED,
          false /*must*/,
          0 /*mapper_id*/,
          get_std_hash(view) /*MappingTagID*/);
      FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
      fm.wait_all_results();
      int idx = 0;
      Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
      ncclComm_t *nccl_comms =
          (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
      for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
        nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
      }
      view_hash_to_nccl_comms[get_std_hash(view)] = nccl_comms;
    }
  }
}

void FFModel::optimize_unnecessary_gradient_calculations() {
  // If an operator's input is training data
  // No need to compute its gradients
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != nullptr);
      if (op->inputs[i]->owner_op->op_type == OP_INPUT) {
        op->trainableInputs[i] = false;
      }
    }
  }
}

void FFModel::print_operator_regions() const {
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%d)\n", i, operators[i]->op_type);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}

void FFModel::create_label_tensor(LossType loss_type) {
  Op const *final_operator = get_final_operator();

  std::vector<ParallelDim> p_dims = final_operator->outputs[0]->get_shape().dims;

  std::vector<size_t> dims;
  // FIXME: Currently assume 1st input for 1st operator = batch_size
  for (ParallelDim const &dim : p_dims) {
    if (!dim.is_replica_dim) {
      dims.push_back(dim.size);
    }
  }

  DataType label_type = DT_FLOAT;
  if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assign dims[num_dims-1] = 1 for sparse categorical labels
    assert(p_dims[0].degree == 1);
    p_dims[0].size = 1;
    dims[0] = 1;
    label_type = DT_INT32;
  }

  LegionParallelTensorShape label_p_shape = { p_dims, label_type };
  LegionTensorShape label_shape = { dims, label_type };

  // create label tensor
  label_tensor = create_tensor(label_shape, NULL, 0 /*idx*/, false /*create_grad*/);   
  parallel_label_tensor = create_parallel_tensor(label_p_shape);                                       
  label_tensor.value()->parallel_tensor = parallel_label_tensor;                     
  parallel_label_tensor.value()->machine_view =                                      
      final_operator->outputs[0]->machine_view;                              
  map_tensor(parallel_label_tensor.value(), 
             parallel_label_tensor.value()->owner_op,
             this->config.legion_config,
             this->index_space_mgr);
}

void FFModel::populate_tensor_to_parallel_tensor_mapping() {
  for (auto const &layer : layers) {
    // map inputs to parallel tensor
    if (layer->op_type == OP_INPUT) {
      Tensor tensor = layer->outputs[0];
      optional<ParallelTensor> parallel_tensor;
      for (Op const *op : operators) {
        if (op->op_type == OP_INPUT) {
          NoOp *noop = (NoOp *)op;
          if (noop->input_tensor_guid == tensor->tensor_guid) {
            parallel_tensor = op->outputs[0];
          }
        }
      }
      tensor->parallel_tensor = parallel_tensor.value();
    }
    // map weights to parallel_tensor
    assert (layer->weights.size() == layer->numWeights);

    bool found = false;
    for (Op const *op : operators) {
      if (op->layer_guid == layer->layer_guid) {
        found = true;
        assert(op->op_type == layer->op_type);
        assert(op->numWeights == layer->numWeights);
        for (int i = 0; i < layer->numWeights; i++) {
          Tensor weight = layer->weights[i];
          weight->parallel_tensor = op->weights[i];
        }
      }
    }
    assert (found);
  }
}

void FFModel::execute_graph_optimize() {
  FFModel *model = this;
  Context ctx = config.legion_config.lg_ctx;
  Runtime *runtime = config.legion_config.lg_hlr;
  TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID,
                        TaskArgument(&model, sizeof(FFModel *)));
  Future future = runtime->execute_task(ctx, launcher);

  PCG::GraphOptimalViewSerialized ret =
      future.get_result<PCG::GraphOptimalViewSerialized>();
  Deserializer dez(ret.data, ret.total_bytes);
  // Reconstruct operators
  PCG::Graph *best_graph = new PCG::Graph(this);
  std::unordered_map<PCG::Node, MachineView> optimal_views;
  deserialize_graph_optimal_view(dez, best_graph, optimal_views);
  operators.clear();
  convert_graph_to_operators(best_graph, optimal_views);
  best_graph->print_dot();
  delete best_graph;

  this->populate_tensor_to_parallel_tensor_mapping();
}

void FFModel::compile(LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  if (metrics_input == -1) {
    metrics_input = operators.size() - 1;
  }
  Context ctx = config.legion_config.lg_ctx;
  Runtime *runtime = config.legion_config.lg_hlr;
  config.computationMode = comp_mode;
  // if (config.import_strategy_file.length() > 0) {
  //   load_strategies_from_file(config.import_strategy_file,
  //   config.strategies);
  // }
  //  Construct operators from layers
  if (config.only_data_parallel) {
    fprintf(stderr,
            "Note: only_data_parallel is specified, FlexFlow compiles a "
            "data-parallel PCG.\n");
  }
  this->create_operators_from_layers();
  
  // Launch the graph optimize task
  this->execute_graph_optimize();

  bool repl_labels = (operators[operators.size() - 1]->op_type == OP_AGG_SPEC);
  loss_op = {loss_type, repl_labels};
  metrics_op = {loss_type, metrics};

  // Init performance metrics
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID,
                        TaskArgument(&metrics_op.value(), sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);

  if (config.enable_inplace_optimizations) {
    this->perform_inplace_optimizations();
  }

  for (Op *op : this->operators) {
    for (ParallelTensor const &input : op->inputs) {
      assert(input->owner_op != NULL);
    }

    for(ParallelTensor const &weight : op->weights) {
      assert(weight->owner_op != NULL);
      assert(weight->region != LogicalRegion::NO_REGION);
      parameters.push_back(weight);
    }

    op->map_output_tensors(*this);

    if (op->is_parallel_op()) {
      ((ParallelOp *)op)->create_input_partition(*this);
    }
  }

  // Check correctness
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->parallel_tensor_guid != 0);
    }
  }

  this->optimize_unnecessary_gradient_calculations();

  if (config.perform_fusion) {
    this->perform_fusion_optimizations();
  }

  Op *final_operator = get_final_operator();
  // FIXME: currently assume the final operator has exactly one output
  assert(final_operator->numOutputs == 1);
  this->print_operator_regions();

  this->create_label_tensor(loss_type);
  
  // init optimizer
  assert(optimizer != NULL);
  optimizer->init();

#ifdef FF_USE_NCCL
  if (config.computationMode == COMP_MODE_TRAINING) {
    this->initialize_nccl_communicators();
  }
#endif
}

void FFModel::zero_gradients(void) {
  for (int l = operators.size() - 1; l >= 0; l--) {
    operators[l]->zero_grad(*this);
  }
}

std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
    FFModel::get_bwd_edge_map() const {
  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> bwd_edge_map;
  for (auto const &op : this->operators) {
    for (int i = 0; i < op->numInputs; i++) {
      Op *src = (Op *)op->inputs[i]->owner_op;
      bwd_edge_map[src].push_back({op, op->inputs[i]->get_volume()});
    }
  }

  return bwd_edge_map;
};

PerfMetrics
    FFModel::update_metrics_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  Metrics *m = (Metrics *)task->args;
  if (task->futures.size() == 0) {
    // Create an empty future
    PerfMetrics perf;
    return perf;
  }
  assert(task->futures.size() > 1);
  PerfMetrics all_metrics = task->futures[0].get_result<PerfMetrics>();
  for (size_t i = 1; i < task->futures.size(); i++) {
    PerfMetrics one_metrics = task->futures[i].get_result<PerfMetrics>();
    all_metrics.update(one_metrics);
  }
  all_metrics.print(m);
  return all_metrics;
}

void Op::prefetch(FFModel const &ff) {
  // TODO: perform prefetch for performance imporvement
}

// ========================================================
// class FFIterationConfig
// ========================================================
FFIterationConfig::FFIterationConfig() {
  seq_length = -1;
}

void FFIterationConfig::reset() {
  seq_length = -1;
}

};
