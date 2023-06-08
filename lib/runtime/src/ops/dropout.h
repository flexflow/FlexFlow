#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "op-attrs/ops/dropout.h"
#include "op_task_invocation.h"
#include "sim_environment.h"
#include "tasks.h"

namespace FlexFlow {

template <>
void register_task<DROPOUT_INIT_TASK_ID>();
template <>
void register_task<DROPOUT_FWD_TASK_ID>();
template <>
void register_task<DROPOUT_BWD_TASK_ID>();

OpTaskInvocation init(DropoutAttrs const &);
OpTaskInvocation forward(DropoutAttrs const &);
OpTaskInvocation backward(DropoutAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  DropoutAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Dropout : public Op { */
/* public: */
/*   Dropout(FFModel &model, */
/*           const ParallelTensor input, */
/*           float rate, */
/*           unsigned long long seed, */
/*           char const *name); */
/*   Dropout(FFModel &model, Dropout const &other, const ParallelTensor input);
 */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */

/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */

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

/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */

/* public: */
/*   float rate; */
/*   unsigned long long seed; */
/* }; */

} // namespace FlexFlow

#endif
