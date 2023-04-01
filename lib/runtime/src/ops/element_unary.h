#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "layer.h"
#include "operator.h"

namespace FlexFlow {

class ElementUnary : public Op {
public:
  ElementUnary(FFModel &model,
               OperatorType type,
               const ParallelTensor x,
               bool inplace,
               char const *name,
               float scalar);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void map_output_tensors(FFModel &model) override;
  bool can_inplace_output() override;
  bool has_inplace_output() override;
  void do_inplace_output() override;
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
  template <typename T>
  static void
      forward_task_with_type(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  template <typename T>
  static void backward_task_with_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

private:
  bool inplace;

public:
  float scalar;
};

}

#endif 
