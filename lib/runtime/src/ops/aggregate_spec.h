#ifndef _FLEXFLOW_AGGREGATE_SPEC_H_
#define _FLEXFLOW_AGGREGATE_SPEC_H_

#include "op-attrs/ops/aggregate_spec.h"
#include "sim_environment.h"
#include "task_spec/op_task_signature.h"

namespace FlexFlow {

template <>
void register_task<AGG_SPEC_INIT_TASK_ID>();
template <>
void register_task<AGG_SPEC_FWD_TASK_ID>();
template <>
void register_task<AGG_SPEC_BWD_TASK_ID>();

OpTaskInvocation init(AggregateSpecAttrs const &);
OpTaskInvocation forward(AggregateSpecAttrs const &);
OpTaskInvocation backward(AggregateSpecAttrs const &);

CostMetrics
    measure_operator_cost(SimEnvironment const &sim,
                          AggregateSpecAttrs const &,
                          ParallelTensorShape const &gate_preds,
                          ParallelTensorShape const &gate_assign,
                          std::vector<ParallelTensorShape> const &exp_preds,
                          ProfilingSettings const &settings,
                          MachineView const &mv);

} // namespace FlexFlow
#endif
