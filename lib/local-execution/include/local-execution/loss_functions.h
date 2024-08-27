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

#ifndef _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOSS_FUNCTIONS_H_
#define _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOSS_FUNCTIONS_H_

#include "local-execution/task_impl_function.dtg.h"
#include "local-execution/task_invocation.h"
#include "local-execution/task_signature.h"
#include "op-attrs/ops/loss_functions.h"

namespace FlexFlow {

TaskImplFunction get_loss_bwd_task_impl();
TaskSignature get_loss_bwd_signature();
TaskInvocation
    backward(LossAttrs const &, tensor_guid_t logit, tensor_guid_t label);

} // namespace FlexFlow

#endif
