#ifndef _FLEXFLOW_REDUCTION_H
#define _FLEXFLOW_REDUCTION_H

#include "op-attrs/ops/combine.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<REDUCTION_INIT_TASK_ID>();
template <>
void register_task<REDUCTION_FWD_TASK_ID>();
template <>
void register_task<REDUCTION_BWD_TASK_ID>();

OpTaskInvocation init(ReductionAttrs const &);
OpTaskInvocation forward(ReductionAttrs const &);
OpTaskInvocation backward(ReductionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReductionAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Reduction : public ParallelOp { */
/* public: */
/*   Reduction(FFModel &model, */
/*             ParallelTensor const &input, */
/*             int reduction_legion_dim, */
/*             int reduction_degree, */
/*             char const *name = NULL); */
/*   Reduction(FFModel &model, */
/*             ReductionAttrs const &attrs, */
/*             std::vector<ParallelTensor> const &inputs, */
/*             char const *name = nullptr); */
/*   void create_input_partition(FFModel &model) override; */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   bool append_parallel_op_info( */
/*       std::vector<ParallelOpInfo> &parallel_ops) const override; */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */

/* public: */
/*   int reduction_dim, reduction_degree; */
/* }; */

} // namespace FlexFlow

#endif
