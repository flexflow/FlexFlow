#ifndef _FLEXFLOW_PARTITION_H
#define _FLEXFLOW_PARTITION_H

#include "flexflow/inference.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_ops/partition_params.h"
#include "parallel_op.h"

namespace FlexFlow {

class Repartition : public ParallelOp {
public:
  using Params = RepartitionParams;
  using Input = ParallelTensor;

  Repartition(FFModel &model,
              const ParallelTensor input,
              int repartition_legion_dim,
              int repartition_degree,
              char const *name = NULL);
  Repartition(FFModel &model,
              Params const &params,
              Input const input,
              char const *name = nullptr);
  void create_input_partition(FFModel &model) override;
  void create_input_partition_inference(
      FFModel &model,
      std::vector<ParallelTensor> const &batch_inputs,
      std::vector<ParallelTensor> const &batch_outputs) override;
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &bc,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void backward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
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
  template <typename T>
  static void
      forward_task_with_type(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  template <typename T>
  static void backward_task_with_type(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  tl::optional<RecordFormatter> as_dot() const override;

  Params get_params() const;

public:
  int repartition_dim, repartition_degree;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_PARTITION_H
