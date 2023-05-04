#ifndef _FLEXFLOW_PLACE_HOLDER_H_
#define _FLEXFLOW_PLACE_HOLDER_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/node.h"
#include "flexflow/ops/place_holder_params.h"

namespace FlexFlow {

class PlaceHolderMeta : public OpMeta {
public:
  PlaceHolderMeta(FFHandler handle);
};

class PlaceHolder : public Op {
public:
  using Params = PlaceHolderParams;
  using Input = ParallelTensor;
  PlaceHolder(FFModel &model,
              const ParallelTensor input,
              LayerID const &_layer_guid,
              char const *name);
  PlaceHolder(FFModel &model,
              PlaceHolder const &other,
              const ParallelTensor input);
  PlaceHolder(FFModel &model,
              Params const &params,
              Input const input,
              char const *name = nullptr);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfig const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
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
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  Params get_params() const;

public:
};

}; // namespace FlexFlow

#endif
