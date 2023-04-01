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
#ifndef _FLEXFLOW_CAST_H
#define _FLEXFLOW_CAST_H

#include "layer.h"
#include "operator.h"

namespace FlexFlow {

class Cast : public Op {
public:
  Cast(FFModel &model,
       ParallelTensor const &input,
       DataType dtype,
       char const *name);
  Cast(FFModel &model,
       Params const &params,
       Input const &input,
       char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static PerDeviceOpState *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename IDT>
  static void backward_task_with_1_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename IDT, typename ODT>
  static void backward_task_with_2_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const;
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
};

}

#endif 
