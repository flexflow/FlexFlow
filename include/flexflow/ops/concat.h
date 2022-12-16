#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/concat_params.h"

namespace FlexFlow {

class Concat : public Op {
public:
  using Params = ConcatParams;
  using Input = std::vector<ParallelTensor>;

  Concat(FFModel &model,
         int n,
         ParallelTensor const *inputs,
         int axis,
         char const *name);
  Concat(FFModel &model,
         ConcatParams const &params,
         std::vector<ParallelTensor> const &inputs,
         char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
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

  Params get_params() const;

public:
  int legion_axis;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_CONCAT_H
