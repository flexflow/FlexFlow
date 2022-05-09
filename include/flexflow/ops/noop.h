/**
 * @file noop.h
 * @brief NoOp Operator
 *
 * @copyright
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

#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "flexflow/model.h"

namespace FlexFlow {

/**
 * @brief The characteristic of NoOp, combined by a vector of inputs and the
 * op_type.
 *
 * This is actually not a "parameter", but some kind of characteristic to
 * distinguish different NoOp objects. To maintain the naming consistency, name
 * this class as "Params".
 */
class NoOpParams {
 public:
  std::vector<ParallelTensor> inputs;
  OperatorType op_type;

  NoOpParams(const std::vector<ParallelTensor> _inputs,
             const OperatorType _op_type)
      : inputs(_inputs), op_type{_op_type} {}

  friend bool operator==(NoOpParams const& lhs, NoOpParams const& rhs) {
    if (lhs.inputs == rhs.inputs && lhs.op_type == rhs.op_type) {
      return true;
    } else {
      return false;
    }
  }
};

class NoOp : public Op {
 public:
  // This is necessary to support get_or_create_node in model.h
  using Params = NoOpParams;

  NoOp(FFModel& model, OperatorType type, const ParallelTensor output,
       const char* name = nullptr);
  NoOp(FFModel& model, OperatorType type, size_t input_tensor_guid,
       const ParallelTensor output, const char* name = nullptr);
  void init(const FFModel&) override;
  void forward(const FFModel&) override;
  void backward(const FFModel&) override;
  void print_layer(const FFModel& model) override { assert(0); }
  bool measure_operator_cost(Simulator* sim, const MachineView& pc,
                             CostMetrics& cost_metrics) const override;
  static OpMeta* init_task(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           Legion::Context ctx, Legion::Runtime* runtime);

  NoOpParams get_params() const;
  tl::optional<RecordFormatter> as_dot() const override;

 public:
  size_t input_tensor_guid;
};

};  // namespace FlexFlow

namespace std {

template <>
struct hash<FlexFlow::NoOpParams> {
  size_t operator()(FlexFlow::NoOpParams const& params) const {
    size_t hash = 0;
    for (const auto& tensor : params.inputs) {
      hash_combine(hash, tensor);
    }
    hash_combine(hash, params.op_type);
    return hash;
  }
};

};  // namespace std

#endif  // _FLEXFLOW_NOOP_H
