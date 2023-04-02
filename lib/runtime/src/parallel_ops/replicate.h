#ifndef _FLEXFLOW_REPLICATE_H
#define _FLEXFLOW_REPLICATE_H

#include "operator.h"
#include "parallel_op.h"

namespace FlexFlow {

class Replicate : public ParallelOp {
public:
  Replicate(FFModel &model,
            ParallelTensor const &input,
            int replicate_legion_dim,
            int replicate_degree,
            char const *name = NULL);
  Replicate(FFModel &model,
            ReplicateAttrs const &attrs,
            std::vector<ParallelTensor> const &inputs,
            char const *name = nullptr);
  void create_input_partition(FFModel &model) override;
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
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
  int replicate_dim, replicate_degree;
};

}

#endif 
