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

#include "ops/noop.h"
#include "hash_utils.h"

using namespace Legion;

NoOp::NoOp(FFModel& model,
           OperatorType _type,
           const Tensor _output,
           const char* _name)
: Op(model, _type, _name, 0/*inputs*/, 0/*weights*/, 1/*outputs*/)
{
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

OpMeta* NoOp::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  FFHandler handle = *((const FFHandler*) task->local_args);
  OpMeta* m = new OpMeta(handle);
  return m;
}

void NoOp::init(const FFModel& ff)
{
  parallel_is = outputs[0]->parallel_is;
  // For OP_INPUT, initialize tensor to zero
  if (op_type == OP_INPUT) {
    ConstantInitializer* initializer = NULL;
    if (outputs[0]->data_type == DT_FLOAT) {
      initializer = new ConstantInitializer(0.0f);
    } else if (outputs[0]->data_type == DT_INT64) {
      initializer = new ConstantInitializer((int64_t)0);
    } else if (outputs[0]->data_type == DT_INT32) {
      initializer = new ConstantInitializer((int)0);
    }
    Runtime* runtime = ff.config.lg_hlr;
    Context ctx = ff.config.lg_ctx;
    ArgumentMap argmap;
    IndexLauncher launcher(CONSTANT_INIT_TASK_ID, parallel_is,
                           TaskArgument(initializer, sizeof(ConstantInitializer)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           outputs[0]->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(outputs[0]->part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else if (op_type == OP_WEIGHT) {
    ArgumentMap argmap;
    Context ctx = ff.config.lg_ctx;
    Runtime* runtime = ff.config.lg_hlr;
    set_argumentmap_for_init(ff, argmap);
    IndexLauncher launcher(NOOP_INIT_TASK_ID, parallel_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           outputs[0]->machine_view.hash());
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
    set_opmeta_from_futuremap(ff, fm);
  }
}

void NoOp::forward(const FFModel& ff)
{}

void NoOp::backward(const FFModel& ff)
{}

bool NoOp::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return true;
}

size_t NoOp::get_params_hash() const {
  size_t hash = 0;
  for (int i = 0; i < this->numInputs; i++) {
    hash_combine(hash, this->inputs[i]);
  }
  hash_combine(hash, this->op_type);

  return hash;
}

Node FFModel::get_or_create_noop_node(const Tensor input)
{
  size_t hash = input->get_owner_independent_hash();
  NoOp* noop = NULL;
  const auto& it = cached_noop_ops.find(hash);
  if (it != cached_noop_ops.end()) {
    noop = it->second;
  } else {
    noop = new NoOp(*this, OP_NOOP, input, NULL);
    cached_noop_ops[hash] = noop;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = noop;
  return ret;
}

Node FFModel::get_or_create_input_node(const TensorShape& output_shape)
{
  size_t hash = std::hash<TensorShape>{}(output_shape); 
  NoOp* input = NULL;
  const auto& it = cached_input_ops.find(hash);
  if (it != cached_input_ops.end()) {
    input = it->second;
  } else {
    Tensor tensor = new TensorBase();
    tensor->ts_guid = tensor_global_guid++;
    tensor->data_type = DT_FLOAT; // TODO FIXME @lockshaw
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
    assert (tensor->check_valid());
    input = new NoOp(*this, OP_INPUT, tensor, NULL);
  }

  return this->new_node(input);
}
