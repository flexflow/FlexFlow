#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "operator.h"
#include "layer.h"

namespace FlexFlow {

class NoOp : public Op {
public:
  NoOp(FFModel &model,
       OperatorType type,
       ParallelTensor const &output,
       char const *name = NULL);
  NoOp(FFModel &model,
       OperatorType type,
       size_t input_tensor_guid,
       ParallelTensor const &output,
       char const *name = NULL);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  static PerDeviceOpState *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);

  tl::optional<RecordFormatter> as_dot() const override;

public:
  size_t input_tensor_guid;
};

}

#endif
