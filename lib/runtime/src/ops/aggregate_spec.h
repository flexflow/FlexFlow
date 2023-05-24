#ifndef _FLEXFLOW_AGGREGATE_SPEC_H_
#define _FLEXFLOW_AGGREGATE_SPEC_H_

#include "op_task_signature.h"
#include "sim_environment.h"
#include "op-attrs/ops/aggregate_spec.h"

namespace FlexFlow {

template <> void register_task<AGG_SPEC_INIT_TASK_ID>();
template <> void register_task<AGG_SPEC_FWD_TASK_ID>();
template <> void register_task<AGG_SPEC_BWD_TASK_ID>();

OpTaskInvocation init(AggregateSpecAttrs const &);
OpTaskInvocation forward(AggregateSpecAttrs const &);
OpTaskInvocation backward(AggregateSpecAttrs const &);

CostMetrics measure_operator_cost(SimEnvironment const &sim, 
                                  AggregateSpecAttrs const &,
                                  ParallelTensorShape const &gate_preds,
                                  ParallelTensorShape const &gate_assign,
                                  std::vector<ParallelTensorShape> const &exp_preds,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);



// class AggregateSpec : public Op {
// public:
//   AggregateSpec(FFModel &model,
//                 ParallelTensor const *inputs,
//                 int _n,
//                 float _lambda_bal,
//                 char const *name);
//   void init(FFModel const &) override;
//   void forward(FFModel const &) override;
//   void backward(FFModel const &) override;
//   static Op *
//       create_operator_from_layer(FFModel &model,
//                                  Layer const *layer,
//                                  std::vector<ParallelTensor> const &inputs);
//   bool measure_operator_cost(Simulator *sim,
//                              MachineView const &pc,
//                              CostMetrics &cost_metrics) const override;
// 
//   OpTaskBinding get_init_task_binding() const override;
//   TaskID get_init_task_id() const override;
//   OpTaskBinding get_fwd_task_binding() const override;
//   TaskID get_fwd_task_id() const override;
//   OpTaskBinding get_bwd_task_binding() const override;
//   TaskID get_bwd_task_id() const override;
// public:
//   AggregateSpecAttrs attrs;
// };

}
#endif
