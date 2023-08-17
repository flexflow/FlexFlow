#ifndef _FLEXFLOW_ALLREDUCE_H
#define _FLEXFLOW_ALLREDUCE_H

#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_ops/allreduce_params.h"
#include "parallel_op.h"

namespace FlexFlow {

class AllReduce : public ParallelOp {
public:
  using Params = AllReduceParams;
  using Input = ParallelTensor;

  AllReduce(FFModel &model,
            const ParallelTensor input,
            int allreduce_legion_dim,
            char const *name = NULL);
  AllReduce(FFModel &model,
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
  static void inference_task(Legion::Task const *task,
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

  Params get_params() const;

public:
  int allreduce_dim;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_ALLREDUCE_H
