/* Copyright 2022 CMU, Stanford, Facebook, LANL
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
#pragma once
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/cast_params.h"

namespace FlexFlow {

class CastMeta : public OpMeta {
public:
  CastMeta(FFHandler handle);
  DataType input_data_type, output_data_type;
};

class Cast : public Op {
public:
  using Params = CastParams;
  using Input = ParallelTensor;
  Cast(FFModel &model,
       ParallelTensor const &input,
       DataType dtype,
       char const *name);
  Cast(FFModel &model,
       Params const &params,
       Input const &input,
       char const *name = nullptr);
  void init(FFModel const &);
  void forward(FFModel const &);
  void backward(FFModel const &);
  void print_layer(FFModel const &model) {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  template <typename IDT>
  static void forward_task_with_1_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  template <typename IDT, typename ODT>
  static void forward_task_with_2_type(
      Legion::Task const *task,
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
  template <typename IDT, typename ODT>
  static void forward_kernel(const IDT *input_ptr,
                             ODT *output_ptr,
                             size_t volume,
                             ffStream_t stream);
  template <typename IDT, typename ODT>
  static void forward_kernel_wrapper(const IDT *input_ptr,
                                     ODT *output_ptr,
                                     size_t volume);
  template <typename IDT, typename ODT>
  static void backward_kernel(const IDT *src_ptr,
                              ODT *dst_ptr,
                              size_t volume,
                              ffStream_t stream);
  template <typename IDT, typename ODT>
  static void
      backward_kernel_wrapper(const IDT *src_ptr, ODT *dst_ptr, size_t volume);
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
  Params get_params() const;
};

}; // namespace FlexFlow
