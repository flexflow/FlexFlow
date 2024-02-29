#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/element_binary_params.h"

namespace FlexFlow {

class ElementBinary : public Op {
public:
  using Params = ElementBinaryParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;

  ElementBinary(FFModel &model,
                OperatorType type,
                const ParallelTensor x,
                const ParallelTensor y,
                bool inplace_a,
                char const *name);
  ElementBinary(FFModel &model,
                Params const &params,
                Input const &inputs,
                char const *name = nullptr,
                bool inplace_a = false);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  void map_output_tensors(FFModel &model) override;
  bool can_inplace_output() override;
  bool has_inplace_output() override;
  void do_inplace_output() override;
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
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Params get_params() const;

public:
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
  int batch_size;
};

}; // namespace FlexFlow

#endif // _FLEXFFLOW_ELEMENT_BINARY_H
