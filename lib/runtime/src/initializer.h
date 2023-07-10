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

#ifndef _FLEXFLOW_INITIALIZER_H_
#define _FLEXFLOW_INITIALIZER_H_

#include "kernels/accessor.h"
#include "legion.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/initializer.h"
#include "pcg/parallel_tensor.h"
#include "runtime/config.h"
#include "task_spec/task_invocation.h"
#include "task_spec/task_signature.h"
#include "tasks.h"

namespace FlexFlow {

struct parallel_tensor_guid_t;

template <>
void register_task<GLOROT_INIT_TASK_ID>();
template <>
void register_task<ZERO_INIT_TASK_ID>();
template <>
void register_task<UNIFORM_INIT_TASK_ID>();
template <>
void register_task<NORMAL_INIT_TASK_ID>();
template <>
void register_task<CONSTANT_INIT_TASK_ID>();

TaskInvocation apply_initializer(GlorotUniform const &,
                                 parallel_tensor_guid_t const &,
                                 ParallelTensor const &,
                                 TensorShape const &);
TaskInvocation apply_initializer(ZeroInitializer const &,
                                 parallel_tensor_guid_t const &,
                                 ParallelTensor const &);
TaskInvocation apply_initializer(UniformInitializer const &,
                                 parallel_tensor_guid_t const &,
                                 ParallelTensor const &);
TaskInvocation apply_initializer(NormInitializer const &,
                                 parallel_tensor_guid_t const &,
                                 ParallelTensor const &);
TaskInvocation apply_initializer(ConstantInitializer const &,
                                 parallel_tensor_guid_t const &,
                                 ParallelTensor const &);

} // namespace FlexFlow

#endif
