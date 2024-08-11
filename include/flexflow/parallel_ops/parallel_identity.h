#ifndef _FLEXFLOW_PARALLEL_IDENTITY_H
#define _FLEXFLOW_PARALLEL_IDENTITY_H

#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_ops/parallel_identity_params.h"
#include "parallel_op.h"

namespace FlexFlow {

class ParallelIdentity : public ParallelOp {
public:
  using Params = ParallelIdentityParams;
  using Input = ParallelTensor;

  ParallelIdentity(FFModel &model,
                   const ParallelTensor input,
                   int parallel_identity_legion_dim,
                   char const *name = NULL);
  ParallelIdentity(FFModel &model,
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
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &bc,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  Legion::FutureMap peft_bwd(FFModel const &,
                             BatchConfigFuture const &bc,
                             std::vector<ParallelTensor> const &,
                             std::vector<ParallelTensor> const &,
                             MachineView const *mv = nullptr) override;
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
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  Params get_params() const;

public:
  int parallel_identity_dim;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_PARALLEL_IDENTITY_H
