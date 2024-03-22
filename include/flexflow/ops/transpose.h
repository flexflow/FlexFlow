#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "flexflow/model.h"
#include "flexflow/ops/transpose_params.h"

namespace FlexFlow {

class TransposeMeta;

class Transpose : public Op {
public:
  using Params = TransposeParams;
  using Input = ParallelTensor;
  Transpose(FFModel &model,
            Params const &params,
            const Input input,
            char const *name = nullptr);
  Transpose(FFModel &model,
            const ParallelTensor input,
            std::vector<int> const &perm,
            char const *name = nullptr);
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
  void init_meta(TransposeMeta *m,
                 Legion::Domain const &in_domain,
                 Legion::Domain const &out_domain) const;
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
  Params get_params() const;

public:
  int perm[MAX_TENSOR_DIM];
};

}; // namespace FlexFlow

#endif
