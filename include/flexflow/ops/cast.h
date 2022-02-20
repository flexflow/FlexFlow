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
#include "flexflow/model.h"

namespace FlexFlow {

class CastMeta : public OpMeta {
public:
  CastMeta(FFHandler handle);
  DataType input_data_type, output_data_type;
};

class Cast : public Op {
public:
  Cast(FFModel& model,
       const ParallelTensor& input,
       DataType dtype,
       const char* name);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  static Op* create_operator_from_layer(
      FFModel& model,
      const Layer* layer,
      const std::vector<ParallelTensor>& inputs);
  static OpMeta* init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
  template<typename IDT>
  static void forward_task_with_1_type(const Legion::Task *task,
                                       const std::vector<Legion::PhysicalRegion> &regions,
                                       Legion::Context ctx, Legion::Runtime *runtime);
  template<typename IDT, typename ODT>
  static void forward_task_with_2_type(const Legion::Task *task,
                                       const std::vector<Legion::PhysicalRegion> &regions,
                                       Legion::Context ctx, Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx, Legion::Runtime *runtime);
  template<typename IDT>
  static void backward_task_with_1_type(const Legion::Task *task,
                                        const std::vector<Legion::PhysicalRegion> &regions,
                                        Legion::Context ctx, Legion::Runtime *runtime);
  template<typename IDT, typename ODT>
  static void backward_task_with_2_type(const Legion::Task *task,
                                        const std::vector<Legion::PhysicalRegion> &regions,
                                        Legion::Context ctx, Legion::Runtime *runtime);
  template<typename IDT, typename ODT>
  static void forward_kernel(const IDT* input_ptr,
                             ODT* output_ptr,
                             size_t volume,
                             ffStream_t stream);
  template<typename IDT, typename ODT>
  static void forward_kernel_wrapper(const IDT* input_ptr,
                                     ODT* output_ptr,
                                     size_t volume);
  template<typename IDT, typename ODT>
  static void backward_kernel(const IDT* src_ptr,
                              ODT* dst_ptr,
                              size_t volume,
                              ffStream_t stream);
  template<typename IDT, typename ODT>
  static void backward_kernel_wrapper(const IDT* src_ptr,
                                      ODT* dst_ptr,
                                      size_t volume);
  bool measure_operator_cost(Simulator* sim,
                             const MachineView& pc,
                             CostMetrics& cost_metrics) const;
};

}; // namespace FlexFlow
