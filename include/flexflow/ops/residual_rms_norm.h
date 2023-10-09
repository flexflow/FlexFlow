#ifndef _FLEXFLOW_RESIDUAL_RMS_NORM_H
#define _FLEXFLOW_RESIDUAL_RMS_NORM_H

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/ops/residual_rms_norm_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class ResidualRMSNormMeta;

class ResidualRMSNorm : public Op {
public:
  using Params = ResidualRMSNormParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;
  ResidualRMSNorm(FFModel &model,
                  LayerID const &_layer_guid,
                  const ParallelTensor _input1,
                  const ParallelTensor _input2,
                  float _eps,
                  int dim,
                  bool allocate_weights,
                  char const *name);
  ResidualRMSNorm(FFModel &model,
                  ResidualRMSNormParams const &params,
                  Input const &inputs,
                  bool allocate_weights,
                  char const *name = nullptr);

  ResidualRMSNorm(FFModel &model,
                  ResidualRMSNorm const &other,
                  Input const &inputs,
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
  ResidualRMSNormParams get_params() const;

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
#endif // _FLEXFLOW_RESIDUAL_RMS_NORM_H
