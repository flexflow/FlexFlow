#pragma once

#include "flexflow/model.h"

namespace FlexFlow {

class Mean : public Op {
public:
  Mean(FFModel &model,
       const ParallelTensor input,
       const std::vector<int> &dims,
       bool keepdims,
       const char *name);
  void init(const FFModel &);
  void forward(const FFModel &);
  void backward(const FFModel &);
  void print_layer(const FFModel &model) { assert(0); }

  static OpMeta *init_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(const Legion::Task *task,
                            const std::vector<Legion::PhysicalRegion> &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             const MachineView &pc,
                             CostMetrics &cost_metrics) const;
};

}; // namespace FlexFlow
