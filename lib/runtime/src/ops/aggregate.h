#ifndef _FLEXFLOW_AGGREGATE_H_
#define _FLEXFLOW_AGGREGATE_H_

#include "operator.h"
#include "layer.h"

namespace FlexFlow {

class Aggregate : public Op {
public:
  Aggregate(FFModel &model,
            ParallelTensor const *inputs,
            int _n,
            float _lambda_bal,
            char const *name);
  Aggregate(FFModel &model,
            ParallelTensor const *inputs, 
            AggregateAttrs const &attrs,
            char const *name);
  Aggregate(FFModel &model,
            Aggregate const &other,
            std::vector<ParallelTensor> const &inputs);
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
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;

  OpTaskBinding get_init_task_binding() const override;
  TaskID get_init_task_id() const override;
  OpTaskBinding get_fwd_task_binding() const override;
  TaskID get_fwd_task_id() const override;
  OpTaskBinding get_bwd_task_binding() const override;
  TaskID get_bwd_task_id() const override;
public:
  AggregateAttrs attrs;
};

template <> OpTaskSignature get_signature<AGGREGATE_INIT_TASK_ID>();
template <> OpTaskSignature get_signature<AGGREGATE_FWD_TASK_ID>();
template <> OpTaskSignature get_signature<AGGREGATE_BWD_TASK_ID>();

}

#endif
