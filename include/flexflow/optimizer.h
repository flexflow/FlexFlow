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

#include "flexflow/parallel_tensor.h"
#include "legion.h"
#include "accessor.h"

namespace FlexFlow {

class FFModel;
class OpMeta;

class Optimizer {
public:
  Optimizer(FFModel const *_model);
  virtual void init(void) = 0;
  virtual void next(void) = 0;
  virtual void update(const ParallelTensor p) = 0;
  virtual void unified_update(std::vector<ParallelTensor> const parameters) = 0;
  FFModel const *model;
};

class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(FFModel const *_model,
               double lr = 0.01f,
               double momentum = 0.0f,
               bool nesterov = false,
               double weight_decay = 0.0f);
  void init(void);
  void next(void);
  void update(const ParallelTensor p);
  void unified_update(std::vector<ParallelTensor> const parameters);
  void set_weight_decay(double _weight_decay);
  static void ps_update_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void ps_update_task_gpu(SGDOptimizer const *op,
                                 float const *w_grad_ptr,
                                 size_t size,
                                 int num_replicas,
                                 float *w_ptr,
                                 float *v_ptr);
#ifdef FF_USE_NCCL
  static void
      nccl_update_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      nccl_unified_update_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);                     
  static void nccl_update_task_gpu(SGDOptimizer const *op,
                                   OpMeta const *meta,
                                   float const *w_grad_ptr,
                                   size_t size,
                                   float *w_ptr,
                                   float *v_ptr);
#endif
  double lr, momentum;
  bool nesterov;
  double weight_decay;
  ParameterSyncType comm_type;
  std::map<Legion::LogicalRegion, ParallelTensor> v_values;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(FFModel const *_model,
                double _alpha = 0.001f,
                double _beta1 = 0.9f,
                double _beta2 = 0.999f,
                double _weight_decay = 0.0f,
                double _epsilon = 1e-8);
  void init(void);
  void next(void);
  void update(const ParallelTensor p);
  void unified_update(std::vector<ParallelTensor> const parameters);
  void set_weight_decay(double _weight_decay);
  static void ps_update_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void ps_update_task_gpu(AdamOptimizer const *op,
                                 float const *w_grad_ptr,
                                 size_t size,
                                 int num_replicas,
                                 float *w_ptr,
                                 float *v_ptr,
                                 float *m_ptr);
#ifdef FF_USE_NCCL
  static void
      nccl_update_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      nccl_unified_update_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);                     
  static void nccl_update_task_gpu(AdamOptimizer const *op,
                                   OpMeta const *meta,
                                   float const *w_grad_ptr,
                                   size_t size,
                                   float *w_ptr,
                                   float *v_ptr,
                                   float *m_ptr);
  static void nccl_unified_update_task_gpu(AdamOptimizer const *op,
                                   OpMeta const *meta,
                                   GenericTensorAccessorR *accWGrads,
                                   size_t *size,
                                   GenericTensorAccessorW *accWs,
                                   GenericTensorAccessorW *accVs,
                                   GenericTensorAccessorW *accMs);                                 
#endif
  double alpha, beta1, beta2, weight_decay, epsilon;
  double alpha_t, beta1_t, beta2_t;
  std::map<Legion::LogicalRegion, ParallelTensor> v_values, m_values;
  size_t reservedWorkSpaceSize = 0;
  int parameters_num = 0;
};

}; // namespace FlexFlow
#endif
