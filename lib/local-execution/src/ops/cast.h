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

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/cast_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(CastAttrs const &);

TaskImplFunction get_cast_fwd_task_impl();
TaskImplFunction get_cast_bwd_task_impl();

OpTaskSignature get_cast_fwd_signature();
OpTaskSignature get_cast_bwd_signature();

OpTaskInvocation forward(CastAttrs const &);
OpTaskInvocation backward(CastAttrs const &);

} // namespace FlexFlow

#endif
