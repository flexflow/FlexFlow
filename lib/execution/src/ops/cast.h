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
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

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

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  CastAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);

} // namespace FlexFlow

#endif
