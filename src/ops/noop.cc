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
  numOutputs = 1;
  outputs[0] = _output;
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
    CostMetrics& cost_metrics)
{
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return true;
}
