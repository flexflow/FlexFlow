#ifndef _FLEXFLOW_REPLICATE_H
#define _FLEXFLOW_REPLICATE_H

#include "op-attrs/ops/replicate.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<REPLICATE_INIT_TASK_ID>();
template <>
void register_task<REPLICATE_FWD_TASK_ID>();
template <>
void register_task<REPLICATE_BWD_TASK_ID>();

OpTaskInvocation init(ReplicateAttrs const &);
OpTaskInvocation forward(ReplicateAttrs const &);
OpTaskInvocation backward(ReplicateAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReplicateAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Replicate : public ParallelOp { */
/* public: */
/*   Replicate(FFModel &model, */
/*             ParallelTensor const &input, */
/*             int replicate_legion_dim, */
/*             int replicate_degree, */
/*             char const *name = NULL); */
/*   Replicate(FFModel &model, */
/*             ReplicateAttrs const &attrs, */
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
/*   int replicate_dim, replicate_degree; */
/* }; */

} // namespace FlexFlow

#endif
