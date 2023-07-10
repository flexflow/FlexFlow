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

#include "op-attrs/ops/cast.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<CAST_INIT_TASK_ID>();
template <>
void register_task<CAST_FWD_TASK_ID>();
template <>
void register_task<CAST_BWD_TASK_ID>();

OpTaskInvocation init(CastAttrs const &);
OpTaskInvocation forward(CastAttrs const &);
OpTaskInvocation backward(CastAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  BatchNormAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Cast : public Op { */
/* public: */
/*   Cast(FFModel &model, */
/*        ParallelTensor const &input, */
/*        DataType dtype, */
/*        char const *name); */
/*   Cast(FFModel &model, */
/*        CastAttrs const &params, */
/*        std::vector<ParallelTensor> const &input, */
/*        char const *name = nullptr); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */
/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   OpTaskBinding get_init_task_binding() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */

/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const; */
/* }; */

} // namespace FlexFlow

#endif
