#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "operator.h"
#include "layer.h"
#include "kernels/topk_kernels.h"

namespace FlexFlow {

class TopK : public Op {
public:
  TopK(FFModel &model,
       ParallelTensor const &input,
       int k,
       bool sorted,
       char const *name);
  TopK(FFModel &model, TopK const &other, ParallelTensor const &input);
  TopK(FFModel &model,
       TopKAttrs const &attrs,
       std::vector<ParallelTensor> const &input,
       char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
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
  void serialize(Legion::Serializer &s) const override;
  /* static PCG::Node deserialize(FFModel &ff, */
  /*                              Legion::Deserializer &d, */
  /*                              ParallelTensor inputs[], */
  /*                              int num_inputs); */
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

public:
  int k;
  bool sorted;
};

}

#endif
