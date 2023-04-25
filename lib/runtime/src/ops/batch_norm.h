#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "operator.h"

namespace FlexFlow {

class BatchNorm : public Op {
public:
  BatchNorm(FFModel &model,
            const ParallelTensor input,
            const ParallelTensor scale,
            const ParallelTensor bias,
            bool relu,
            char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void update(FFModel const &);

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
                             CostMetrics &cost_metrics) const override;
  OpTaskBinding get_init_task_binding() const override;
  OpTaskBinding get_fwd_task_binding() const override;
  OpTaskBinding get_bwd_task_binding() const override;
public:
  bool relu;
  int num_replica;
};


}

#endif
