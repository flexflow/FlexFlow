#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H

#include "operator.h"
#include "layer.h"
#include "layer_id.h"

namespace FlexFlow {

class LayerNormMeta;

class LayerNorm : public Op {
public:
  LayerNorm(FFModel &model,
            LayerNormAttrs const &attrs,
            ParallelTensor input,
            char const *name = nullptr,
            bool allocate_weights = false);
  LayerNorm(FFModel &model,
            LayerID const &layer_guid,
            ParallelTensor const &input,
            std::vector<int> const &axes,
            bool _elementwise_affine,
            float _eps,
            bool allocate_weights,
            char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  /* void serialize(Legion::Serializer &) const override; */
  /* static PCG::Node deserialize(FFModel &ff, */
  /*                              Legion::Deserializer &d, */
  /*                              ParallelTensor inputs[], */
  /*                              int num_inputs); */
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;

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

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  std::vector<int> axes;
};

}

#endif
