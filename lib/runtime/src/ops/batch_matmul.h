#ifndef _FLEXFLOW_BATCH_MATMUL_H
#define _FLEXFLOW_BATCH_MATMUL_H

#include "operator.h"
#include "layer.h"

namespace FlexFlow {

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel &model,
              const ParallelTensor A,
              const ParallelTensor B,
              int a_seq_length_dim,
              int b_seq_length_dim,
              char const *name = nullptr);
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void serialize(Legion::Serializer &) const override;
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

  OpTaskBinding get_init_task_binding() const override;
  OpTaskBinding get_fwd_task_binding() const override;
  OpTaskBinding get_bwd_task_binding() const override;
private:
  template <int NDIM>
  void init_with_dim(FFModel const &ff);
  template <int NDIM>
  void forward_with_dim(FFModel const &ff);
  template <int NDIM>
  void backward_with_dim(FFModel const &ff);

public:
  int a_seq_length_dim, b_seq_length_dim;
};

}; // namespace FlexFlow

#endif
