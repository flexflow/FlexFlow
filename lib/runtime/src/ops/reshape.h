#ifndef _FLEXFLOW_RESHAPE_H
#define _FLEXFLOW_RESHAPE_H

#include "layer.h"
#include "operator.h"

namespace FlexFlow {

class Reshape : public Op {
public:
  Reshape(FFModel &model,
          ParallelTensor const &input,
          std::vector<int> const &shape,
          char const *name);
  Reshape(FFModel &model,
          ReshapeAttrs const &attrs,
          std::vector<ParallelTensor> const &input,
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
  /* static PCG::Node deserialize(FFModel &ff, */
  /*                              Legion::Deserializer &d, */
  /*                              ParallelTensor inputs[], */
  /*                              int num_inputs); */
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

public:
  stack_vector<int, MAX_TENSOR_DIM> shape;
};

}

#endif
