/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
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

#include "flexflow/ops/noop.h"
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
using Legion::LogicalPartition;
using Legion::LogicalRegion;
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

NoOp::NoOp(FFModel &model,
           OperatorType _type,
           const ParallelTensor _output,
           char const *_name)
    : Op(model, _type, _name, 0 /*inputs*/, 0 /*weights*/, 1 /*outputs*/),
      input_tensor_guid(0) {
  // NOOP takes one input and has one output
  // both of them are _output
  if (op_type == OP_NOOP) {
    numInputs = 1;
    inputs[0] = _output;
  }
  outputs[0] = _output;
  outputs[0]->owner_op = this;
  outputs[0]->owner_idx = 0;
}

NoOp::NoOp(FFModel &model,
           OperatorType _type,
           size_t _input_tensor_guid,
           const ParallelTensor _output,
           char const *_name)
    : Op(model, _type, _name, 0 /*inputs*/, 0 /*weights*/, 1 /*outputs*/),
      input_tensor_guid(_input_tensor_guid) {
  // NOOP takes one input and has one output
  // both of them are _output
  if (op_type == OP_NOOP) {
    numInputs = 1;
    inputs[0] = _output;
  }
  outputs[0] = _output;
  outputs[0]->owner_op = this;
  outputs[0]->owner_idx = 0;
}

NoOp::NoOp(FFModel &model,
           Params const &params,
           std::vector<ParallelTensor> const &inputs,
           char const *name)
    : Op(model, params.op_type, name, 0 /*weights*/, 1 /*outputs*/, inputs) {
  if (params.op_type == OP_NOOP) {
    assert(inputs.size() == 1);
    this->inputs[0] = inputs[0];
    this->outputs[0] = inputs[0];
  } else {
    assert(params.op_type == OP_INPUT);
    assert(inputs.size() == 0);
    auto input_metadata = params.input_metadata.value();
    if (mp::holds_alternative<size_t>(input_metadata)) {
      assert (false && "Error: unsupported case in OP_INPUT constructor. Please file an issue with the FlexFlow developers.");
      this->input_tensor_guid = mp::get<size_t>(input_metadata);
    } else {
      ParallelTensor tensor = new ParallelTensorBase();
      tensor->parallel_tensor_guid = model.parallel_tensor_global_guid++;
      tensor->data_type = DT_FLOAT; // TODO FIXME @lockshaw
      ParallelTensorShape output_shape =
          mp::get<ParallelTensorShape>(input_metadata);
      tensor->num_dims = output_shape.num_dims;
      int parallel_idx = 0;
      for (int i = 0; i < output_shape.num_dims; i++) {
        tensor->dims[i].size = output_shape.dims[i].size;
        tensor->dims[i].degree = output_shape.dims[i].degree;
        if (tensor->dims[i].degree > 1) {
          tensor->dims[i].parallel_idx = parallel_idx;
          parallel_idx++;
        } else {
          tensor->dims[i].parallel_idx = -1;
        }
      }
      assert(tensor->check_valid());
      this->outputs[0] = tensor;
    }
  }
  outputs[0]->owner_op = this;
  outputs[0]->owner_idx = 0;
}

OpMeta *NoOp::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  FFHandler handle = *((FFHandler const *)task->local_args);
  OpMeta *m = new OpMeta(handle);
  return m;
}

void NoOp::init(FFModel const &ff) {
  parallel_is = outputs[0]->parallel_is;
  // For OP_INPUT, initialize tensor to zero
  if (op_type == OP_INPUT) {
    assert(outputs[0]->region != LogicalRegion::NO_REGION);
    if (outputs[0]->part == LogicalPartition::NO_PART)
      return;
    ConstantInitializer *initializer = NULL;
    if (outputs[0]->data_type == DT_FLOAT) {
      initializer = new ConstantInitializer(0.0f);
    } else if (outputs[0]->data_type == DT_INT64) {
      initializer = new ConstantInitializer((int64_t)0);
    } else if (outputs[0]->data_type == DT_INT32) {
      initializer = new ConstantInitializer((int)0);
    }
    Runtime *runtime = ff.config.lg_hlr;
    Context ctx = ff.config.lg_ctx;
    ArgumentMap argmap;
    IndexLauncher launcher(
        CONSTANT_INIT_TASK_ID,
        parallel_is,
        TaskArgument(initializer, sizeof(ConstantInitializer)),
        argmap,
        Predicate::TRUE_PRED,
        false /*must*/,
        0 /*mapper_id*/,
        outputs[0]->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else if (op_type == OP_WEIGHT) {
    ArgumentMap argmap;
    Context ctx = ff.config.lg_ctx;
    Runtime *runtime = ff.config.lg_hlr;
    set_argumentmap_for_init(ff, argmap);
    IndexLauncher launcher(NOOP_INIT_TASK_ID,
                           parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           outputs[0]->machine_view.hash());
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
    set_opmeta_from_futuremap(ff, fm);
  }
}

void NoOp::forward(FFModel const &ff) {}

void NoOp::backward(FFModel const &ff) {}

bool NoOp::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return true;
}

/* using PCG::Node; */
/* Node FFModel::get_or_create_noop_node(const ParallelTensor input) { */
/*   size_t hash = input->get_owner_independent_hash(); */
/*   NoOp *noop = NULL; */
/*   auto const &it = cached_noop_ops.find(hash); */
/*   if (it != cached_noop_ops.end()) { */
/*     noop = it->second; */
/*   } else { */
/*     noop = new NoOp(*this, OP_NOOP, input, NULL); */
/*     cached_noop_ops[hash] = noop; */
/*   } */
/*   Node ret; */
/*   ret.guid = node_global_guid++; */
/*   ret.ptr = noop; */
/*   return ret; */
/* } */

/* Node FFModel::get_or_create_input_node( */
/*     ParallelTensorShape const &output_shape) { */
/*   size_t hash = std::hash<ParallelTensorShape>{}(output_shape); */
/*   NoOp *input = NULL; */
/*   auto const &it = cached_input_ops.find(hash); */
/*   if (it != cached_input_ops.end()) { */
/*     input = it->second; */
/*   } else { */
/*     assert(tensor->check_valid()); */
/*     input = new NoOp(*this, OP_INPUT, tensor, NULL); */
/*   } */

/*   return this->new_node(input); */
/* } */

tl::optional<RecordFormatter> NoOp::as_dot() const {
  RecordFormatter rf;
  {
    std::ostringstream oss;
    oss << "shape(" << this->outputs[0]->get_shape() << ")";
    rf << oss.str();
  }
  return rf;
}

NoOpParams NoOp::get_params() const {
  NoOpParams params;
  params.op_type = this->op_type;
  if (this->op_type == OP_INPUT) {
    params.input_metadata = this->input_tensor_guid;
  }

  return params;
}

}; // namespace FlexFlow
