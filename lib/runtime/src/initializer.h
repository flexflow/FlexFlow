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

#include "legion.h"
#include "runtime/config.h"
#include "kernels/accessor.h"
#include "task_signature.h"
#include "task_invocation.h"

namespace FlexFlow {

struct ParallelTensor;
struct parallel_tensor_guid_t;

template <> void register_task<GLOROT_INIT_TASK_ID>();
template <> void register_task<ZERO_INIT_TASK_ID>();
template <> void register_task<UNIFORM_INIT_TASK_ID>();
template <> void register_task<NORMAL_INIT_TASK_ID>();
template <> void register_task<CONSTANT_INIT_TASK_ID>();

class GlorotUniform : public use_visitable_cmp<GlorotUniform> {
public:
  GlorotUniform() = delete;
  GlorotUniform(int seed);
public:
  int seed;
  /* float scale; */
  /* DataType data_type; */
};

class ZeroInitializer : public use_visitable_cmp<ZeroInitializer> {
public:
  ZeroInitializer() = default;
};

class UniformInitializer : public use_visitable_cmp<UniformInitializer> {
public:
  UniformInitializer(int seed, float min, float max);
public:
  int seed;
  float min_val, max_val;
};

class NormInitializer : public use_visitable_cmp<NormInitializer> {
public:
  NormInitializer(int seed, float mean, float stddev);
public:
  int seed;
  float mean, stddev;
};

class ConstantInitializer : public use_visitable_cmp<ConstantInitializer> {
public:
  ConstantInitializer(DataTypeValue const &value);

public:
  DataTypeValue value;
};

using Initializer = variant<
  GlorotUniform,
  ZeroInitializer,
  UniformInitializer,
  NormInitializer,
  ConstantInitializer
>;

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

}

VISITABLE_STRUCT(::FlexFlow::GlorotUniform, seed);
VISITABLE_STRUCT_EMPTY(::FlexFlow::ZeroInitializer);
VISITABLE_STRUCT(::FlexFlow::UniformInitializer, seed, min_val, max_val);
VISITABLE_STRUCT(::FlexFlow::NormInitializer, seed, mean, stddev);
VISITABLE_STRUCT(::FlexFlow::ConstantInitializer, value);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<Initializer>::value, "");
}

#endif
