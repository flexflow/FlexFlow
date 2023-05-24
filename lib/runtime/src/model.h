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


class FFModel {
public:
  FFModel() = delete;
  FFModel(FFConfig const &, 
          ComputationGraph const &, 
          ParallelComputationGraph const &,
          Optimizer const &, 
          RuntimeBacking const &,
          EnableProfiling const &,
          Metrics const &);

  ExecutableTaskInvocation resolve(TaskInvocation const &) const;
  TaskReturnAccessor execute(ExecutableTaskInvocation const &) const;
  TaskReturnAccessor execute(std::vector<ExecutableTaskInvocation> const &);
  std::pair<Legion::TaskArgument, TaskArgumentFormat> construct_legion_task_arg(ExecutableTaskBinding const &);

  void init_operators();
  void forward(int seq_length = -1);  
  void backward(int seq_length = -1);
  void update();

  template <typename T>
  void set_tensor(TensorDims const &, T const *);

  template <typename T>
  void get_tensor(tensor_guid_t, T *data);
  
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void reset_metrics();
  void prefetch();
  void compute_metrics();
  void compile(LossFunction loss_type,
               std::vector<Metric> const &metrics,
               ComputationMode comp_mode = ComputationMode::TRAINING);
  void compile(Optimizer const &optimizer,
               LossFunction loss_type,
               std::vector<Metric> const &metrics,
               ComputationMode comp_mode = ComputationMode::TRAINING);
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();

  // APIs for setting iteration configs
private:
  void execute_graph_optimize();
  void print_operator_regions() const;
  void create_label_tensor(LossFunction);
public:
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer optimizer;
  LossAttrs loss_op;
  Metrics metrics_op;
  /* optional<ParallelTensor> parallel_label_tensor; */
  /* optional<Tensor> label_tensor; */
  SimEnvFactory sim_factory;

  ComputationGraph computation_graph;
  ParallelComputationGraph pcg;
  TensorMapping tensor_map;
  RuntimeBacking runtime_backing;

  EnableProfiling enable_profiling;
};


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
