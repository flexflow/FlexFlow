#ifndef _FLEXFLOW_COMBINE_H
#define _FLEXFLOW_COMBINE_H

#include "op-attrs/ops/combine.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<COMBINE_INIT_TASK_ID>();
template <>
void register_task<COMBINE_FWD_TASK_ID>();
template <>
void register_task<COMBINE_BWD_TASK_ID>();

OpTaskInvocation init(CombineAttrs const &);
OpTaskInvocation forward(CombineAttrs const &);
OpTaskInvocation backward(CombineAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  CombineAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Combine : public ParallelOp { */
/* public: */
/*   Combine(FFModel &model, */
/*           ParallelTensor const &input, */
/*           int combine_legion_dim, */
/*           int combine_degree, */
/*           char const *name = NULL); */
/*   Combine(FFModel &model, */
/*           CombineAttrs const &params, */
/*           std::vector<ParallelTensor> const &input, */
/*           char const *name = nullptr); */
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
/*   template <typename T> */
/*   static void */
/*       forward_task_with_type(Legion::Task const *task, */
/*                              std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                              Legion::Context ctx, */
/*                              Legion::Runtime *runtime); */
/*   template <typename T> */
/*   static void backward_task_with_type( */
/*       Legion::Task const *task, */
/*       std::vector<Legion::PhysicalRegion> const &regions, */
/*       Legion::Context ctx, */
/*       Legion::Runtime *runtime); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &mv, */
/*                              CostMetrics &cost_metrics) const override; */

/*   tl::optional<RecordFormatter> as_dot() const override; */

/* public: */
/*   int combine_dim, combine_degree; */
/* }; */

} // namespace FlexFlow

#endif
