#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "fftype.h"
#include "layer.h"
#include "flexflow/node.h"
#include "op_meta.h"
#include "operator.h"
#include "op-attrs/dropout_params.h"

namespace FlexFlow {

class Dropout : public Op {
public:
  Dropout(FFModel &model,
          const ParallelTensor input,
          float rate,
          unsigned long long seed,
          char const *name);
  Dropout(FFModel &model, Dropout const &other, const ParallelTensor input);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
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

  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

public:
  float rate;
  unsigned long long seed;
};

}; // namespace FlexFlow

#endif
