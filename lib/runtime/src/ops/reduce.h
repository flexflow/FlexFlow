#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H

#include "layer.h"
#include "operator.h"

namespace FlexFlow {

class Reduce : public Op {
public:
  Reduce(FFModel &model,
         const ParallelTensor input,
         std::vector<int> const &axes,
         bool keepdims,
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
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

public:
  int num_axes;
  int axes[MAX_TENSOR_DIM];
  bool keepdims;
};

}

#endif
