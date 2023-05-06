#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "layer.h"
#include "operator.h"
#include "op-attrs/ops/concat.h"

namespace FlexFlow {

class Concat : public Op {
public:
  using Attrs = ConcatAttrs;

  Concat(FFModel &model,
         int n,
         ParallelTensor const *inputs,
         int axis,
         char const *name);
  Concat(FFModel &model,
         Attrs const &attrs,
         std::vector<ParallelTensor> const &inputs,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
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
  int legion_axis;
};

}

#endif
