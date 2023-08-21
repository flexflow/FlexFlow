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

#ifndef _FLEXFLOW_OPTIMIZER_H_
#define _FLEXFLOW_OPTIMIZER_H_

#include "legion.h"
#include "pcg/optimizer.h"
#include "pcg/parallel_tensor.h"
#include "task_spec/task_invocation.h"
#include "tasks.h"

namespace FlexFlow {

template <>
void register_task<PS_PREFETCH_TASK_ID>();
template <>
void register_task<SGD_UPD_PS_TASK_ID>();
template <>
void register_task<SGD_UPD_NCCL_TASK_ID>();
template <>
void register_task<ADAM_UPD_PS_TASK_ID>();
template <>
void register_task<ADAM_UPD_NCCL_TASK_ID>();

/* class Optimizer { */
/* public: */
/*   Optimizer(FFModel const *_model); */
/*   virtual void init(void) = 0; */
/*   virtual void next(void) = 0; */
/*   virtual void update(const ParallelTensor p) = 0; */
/*   FFModel const *model; */
/* }; */

TaskInvocation init(SGDOptimizer const &);
std::vector<TaskInvocation> update(SGDOptimizer const &,
                                   parallel_tensor_guid_t const &,
                                   ParallelTensor const &,
                                   parallel_tensor_guid_t const &sgd_v);

std::vector<TaskInvocation> update(AdamOptimizer const &,
                                   parallel_tensor_guid_t const &,
                                   ParallelTensor const &,
                                   parallel_tensor_guid_t const &adam_m,
                                   parallel_tensor_guid_t const &adam_w);
AdamOptimizer next(AdamOptimizer const &);

using Optimizer = variant<SGDOptimizer, AdamOptimizer>;

/* class SGDOptimizerBacking : public Optimizer { */
/* public: */
/*   SGDOptimizer(FFModel const *_model, */
/*                double lr = 0.01f, */
/*                double momentum = 0.0f, */
/*                bool nesterov = false, */
/*                double weight_decay = 0.0f); */
/*   void init(void); */
/*   void next(void); */
/*   void update(const ParallelTensor p); */
/*   void set_weight_decay(double _weight_decay); */

/*   ParameterSyncType comm_type; */
/*   std::map<Legion::LogicalRegion, ParallelTensor> v_values; */
/* }; */

/* class AdamOptimizerBacking : public Optimizer { */
/* public: */
/*   AdamOptimizer(FFModel const *_model, */
/*                 double _alpha = 0.001f, */
/*                 double _beta1 = 0.9f, */
/*                 double _beta2 = 0.999f, */
/*                 double _weight_decay = 0.0f, */
/*                 double _epsilon = 1e-8); */
/*   void init(void); */
/*   void next(void); */
/*   void update(const ParallelTensor p); */

/* public: */
/*   std::map<Legion::LogicalRegion, ParallelTensor> v_values, m_values; */
/* }; */

} // namespace FlexFlow
#endif
