#pragma once

#include "flexflow/model.h"

namespace FlexFlow {

class ExpertsMeta : public OpMeta {
public:
  ExpertsMeta(FFHandler handler) : OpMeta(handler){};
};

class Experts : public Op {
public:
  using Params = ExpertsParams;
  using Input = std::vector<ParallelTensor>;
  Experts(FFModel &model,
          Params const &params,
          Input const &inputs,
          char const *name = nullptr);
  Experts(FFModel &model,
          ParallelTensor const *inputs,
          int _num_experts,
          int _experts_start_idx,
          int _experts_num_layers,
          int _experts_output_dim_size,
          int _experts_internal_dim_size,
          bool _use_bias,
          ActiMode _activation;
          char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void inference(FFModel const &,
                 std::vector<ParallelTensor> const &,
                 std::vector<ParallelTensor> const &,
                 MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override;
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  Params get_params() const;
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
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  int num_experts;
  int experts_start_idx;
  int experts_num_layers;
  int experts_output_dim_size;
  int experts_internal_dim_size;
  bool use_bias;
  ActiMode activation;
};

}; // namespace FlexFlow
