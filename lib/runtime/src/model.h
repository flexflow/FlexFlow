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
#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_
#include "pcg/machine_specification.h"
#include "runtime/config.h"
#include "legion.h"
#include "metrics_functions.h"
#include "optimizer.h"
#include "recompile.h"
#include <functional>
#include <unistd.h>
#include <utility>
#include "op-attrs/tensor_shape.h"
#include "computation_graph.h"
#include "parallel_computation_graph.h"
#include "tensor_mapping.h"
#include "legion_backing.h"
#include "sim_environment.h"
#include "executable_task_invocation.h"
#include "op-attrs/ops/loss_functions.h"

namespace FlexFlow {

template <> void register_task<FF_INIT_TASK_ID>();

enum ShardingID {
  DataParallelShardingID = 135,
};

MachineView get_basic_data_parallel_machine_view(int num_parts, int dims);
MachineView get_basic_data_parallel_machine_view(FFConfig const &);
MachineView get_basic_data_parallel_machine_view(MachineSpecification const &);

class FFModel {
public:
  FFModel() = delete;
  FFModel(FFConfig const &,
          ComputationGraph const &, 
          ParallelComputationGraph const &,
          Optimizer const &, 
          RuntimeBacking const &,
          EnableProfiling const &,
          Metrics const &,
          SimEnvFactory const &,
          LossAttrs const &,
          TensorMapping const &);

  TaskReturnAccessor execute(ExecutableTaskInvocation const &) const;
  TaskReturnAccessor execute(std::vector<ExecutableTaskInvocation> const &);


  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void compile(LossFunction loss_type,
               std::vector<Metric> const &metrics,
               ComputationMode comp_mode = ComputationMode::TRAINING);
  void compile(Optimizer const &optimizer,
               LossFunction loss_type,
               std::vector<Metric> const &metrics,
               ComputationMode comp_mode = ComputationMode::TRAINING);

  // APIs for setting iteration configs
private:
  void create_label_tensor(LossFunction);
public:
  FFConfig config;
  ComputationGraph computation_graph;
  ParallelComputationGraph pcg;
  Optimizer optimizer;
  RuntimeBacking runtime_backing;
  EnableProfiling enable_profiling;
  Metrics metrics;
  SimEnvFactory sim_factory;
  LossAttrs loss;
  TensorMapping tensor_map;

  FFIterationConfig iter_config;
  /* optional<ParallelTensor> parallel_label_tensor; */
  /* optional<Tensor> label_tensor; */


};

void init_operators(FFModel const &);
void forward(FFModel const &, int seq_length = -1);  
void backward(FFModel const &, int seq_length = -1);
void update(FFModel const &);
void zero_gradients(FFModel const &);
void reset_metrics(FFModel const &);
void compute_metrics(FFModel const &);
void recompile_on_condition(FFModel const &, RecompileState &r);
template <typename T> void set_tensor(FFModel const &, TensorDims const &, T const *);
template <typename T> void get_tensor(FFModel const &, tensor_guid_t, T *data);
  

ExecutableTaskInvocation resolve(TaskInvocation const &, 
                                 EnableProfiling enable_profiling,
                                 RuntimeBacking const &runtime_backing);

void top_level_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void data_load_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void register_custom_tasks();

}

#endif
