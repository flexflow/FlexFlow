#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "op-attrs/ops/batch_norm.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<BATCHNORM_INIT_TASK_ID>();
template <>
void register_task<BATCHNORM_FWD_TASK_ID>();
template <>
void register_task<BATCHNORM_BWD_TASK_ID>();

OpTaskInvocation init(BatchNormAttrs const &);
OpTaskInvocation forward(BatchNormAttrs const &);
OpTaskInvocation backward(BatchNormAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  BatchNormAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class BatchNorm { */
/* public: */
/*   BatchNorm(FFModel &model, */
/*             const ParallelTensor input, */
/*             const ParallelTensor scale, */
/*             const ParallelTensor bias, */
/*             bool relu, */
/*             char const *name); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   void update(FFModel const &); */

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
/*   OpTaskBinding get_init_task_binding() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */
/* public: */
/*   bool relu; */
/*   int num_replica; */
/* }; */

} // namespace FlexFlow

#endif
