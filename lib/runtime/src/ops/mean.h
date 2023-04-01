#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_MEAN_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_MEAN_H

#include "operator.h"

namespace FlexFlow {

class Mean : public Op {
public:
  Mean(FFModel &model,
       const ParallelTensor input,
       std::vector<int> const &dims,
       bool keepdims,
       char const *name);
  void init(FFModel const &);
  void forward(FFModel const &);
  void backward(FFModel const &);
  void print_layer(FFModel const &model) {
    assert(0);
  }

  static PerDeviceOpState *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const;
};

}

#endif
