#pragma once

#include "model.h"

namespace FlexFlow {

class LayerNormMeta;

class LayerNorm : public Op {
public:
  using Params = LayerNormParams;
  using Input = ParallelTensor;
  LayerNorm(FFModel &model,
            LayerNormParams const &params,
            ParallelTensor input,
            char const *name = nullptr,
            bool allocate_weights = false);
  LayerNorm(FFModel &model,
            LayerID const &_layer_guid,
            const ParallelTensor _input,
            std::vector<int> const &axes,
            bool _elementwise_affine,
            float _eps,
            bool allocate_weights,
            char const *name);
  void init(FFModel const &);
  void forward(FFModel const &);
  void backward(FFModel const &);
  void print_layer(FFModel const &model) {
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
  // size_t get_params_hash() const override;
  LayerNormParams get_params() const;

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
                             CostMetrics &cost_metrics) const;

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  std::vector<int> axes;
};

}; // namespace FlexFlow
