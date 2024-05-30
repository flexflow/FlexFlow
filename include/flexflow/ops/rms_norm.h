#ifndef _FLEXFLOW_RMS_NORM_H
#define _FLEXFLOW_RMS_NORM_H

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/ops/rms_norm_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class RMSNormMeta;

class RMSNorm : public Op {
public:
  using Params = RMSNormParams;
  using Input = ParallelTensor;
  RMSNorm(FFModel &model,
          LayerID const &_layer_guid,
          const ParallelTensor _input,
          float _eps,
          int dim,
          bool allocate_weights,
          char const *name);
  RMSNorm(FFModel &model,
          RMSNormParams const &params,
          ParallelTensor input,
          bool allocate_weights,
          char const *name = nullptr);

  RMSNorm(FFModel &model,
          RMSNorm const &other,
          const ParallelTensor input,
          bool allocate_weights);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
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
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  RMSNormParams get_params() const;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  float eps;
  int effective_batch_size;
  int dim, data_dim;
};
} // namespace FlexFlow
#endif // _FLEXFLOW_RMS_NORM_H
