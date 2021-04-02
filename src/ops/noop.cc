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

#include "model.h"

using namespace Legion;

NoOp::NoOp(FFModel& model,
           OperatorType _type,
           const Tensor _output,
           const char* _name)
: Op(model, _type, name, 0/*inputs*/, 0/*weights*/)
{
  // NOOP takes one input and has one output
  // both of them are _output
  if (op_type == OP_NOOP) {
    numInputs = 1;
    inputs[0] = _output;
  }
  numOutputs = 1;
  outputs[0] = _output;
  outputs[0]->owner_op = this;
  outputs[0]->owner_idx = 0;
}

void NoOp::init(const FFModel& ff)
{}

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
