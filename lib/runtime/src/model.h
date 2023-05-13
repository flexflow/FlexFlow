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
#include "accessor.h"
#include "utils/graph/node.h"
#include "op-attrs/operator_attrs.h"
#include "utils/hash-utils.h"
#include "utils/tuple.h"
#include "initializer.h"
#include "layer.h"
#include "legion.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include "optimizer.h"
#include "parallel_tensor.h"
#include "recompile.h"
#include "tensor.h"
#include "tl/optional.hpp"
#include <functional>
#include <unistd.h>
#include <utility>
#include "compiler/compiler.h"
#include "op-attrs/ffconst.h"
#include "layer_id.h"
#include "kernels/ff_handle.h"
#include "op-attrs/tensor_shape.h"
#include "legion_parallel_tensor_shape.h"
#include "index_space_manager.h"
#include "computation_graph.h"
#include "parallel_computation_graph.h"
#include "tensor_mapping.h"
#include "operator.h"
#include "legion_backing.h"
#include "sim_environment.h"
#include "executable_task_invocation.h"

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

  ExecutableTaskInvocation resolve(TaskInvocation const &);
  TaskReturnAccessor execute(ExecutableTaskInvocation const &);
  TaskReturnAccessor execute(std::vector<ExecutableTaskInvocation> const &);

  void init_operators();
  void forward(int seq_length = -1);  
  void backward(int seq_length = -1);
  void update();
  
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  void reset_metrics();
  void prefetch();
  void compute_metrics();
  void compile(LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void compile(Optimizer const &optimizer,
               LossType loss_type,
               std::vector<MetricsType> const &metrics,
               CompMode comp_mode = COMP_MODE_TRAINING);
  void recompile_on_condition(RecompileState &r);
  void zero_gradients();

  // APIs for setting iteration configs
private:
  void execute_graph_optimize();
  void print_operator_regions() const;
  void create_label_tensor(LossType);
public:
  FFConfig config;
  FFIterationConfig iter_config;
  Optimizer optimizer;
  LossAttrs loss_op;
  Metrics metrics_op;
  /* optional<ParallelTensor> parallel_label_tensor; */
  /* optional<Tensor> label_tensor; */
  SimEnvFactory sim_factory;

  IndexSpaceManager index_space_mgr;
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
