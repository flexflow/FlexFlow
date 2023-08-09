#ifndef _FLEXFLOW_PARTITION_H
#define _FLEXFLOW_PARTITION_H

#include "op-attrs/ops/repartition.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<REPARTITION_INIT_TASK_ID>();
template <>
void register_task<REPARTITION_FWD_TASK_ID>();
template <>
void register_task<REPARTITION_BWD_TASK_ID>();

OpTaskInvocation init(RepartitionAttrs const &);
OpTaskInvocation forward(RepartitionAttrs const &);
OpTaskInvocation backward(RepartitionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  RepartitionAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Repartition : public ParallelOp { */
/* public: */
/*   Repartition(FFModel &model, */
/*               ParallelTensor const &input, */
/*               int repartition_legion_dim, */
/*               int repartition_degree, */
/*               char const *name = NULL); */
/*   Repartition(FFModel &model, */
/*               RepartitionAttrs const &attrs, */
/*               std::vector<ParallelTensor> const &inputs, */
/*               char const *name = nullptr); */
/*   void create_input_partition(FFModel &model) override; */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   bool append_parallel_op_info( */
/*       std::vector<ParallelOpInfo> &parallel_ops) const override; */
/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
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

/*   tl::optional<RecordFormatter> as_dot() const override; */

/*   OpTaskBinding get_init_task_binding() const override; */
/*   TaskID get_init_task_id() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   TaskID get_fwd_task_id() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */
/*   TaskID get_bwd_task_id() const override; */
/* public: */
/*   int repartition_dim, repartition_degree; */
/* }; */

} // namespace FlexFlow

#endif
