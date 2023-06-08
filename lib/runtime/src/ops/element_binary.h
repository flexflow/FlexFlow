#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<ELEMENTBINARY_INIT_TASK_ID>();
template <>
void register_task<ELEMENTBINARY_FWD_TASK_ID>();
template <>
void register_task<ELEMENTBINARY_BWD_TASK_ID>();

OpTaskInvocation init(ElementBinaryAttrs const &);
OpTaskInvocation forward(ElementBinaryAttrs const &);
OpTaskInvocation backward(ElementBinaryAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementBinaryAttrs const &attrs,
                                  ParallelTensorShape const &lhs_shape,
                                  ParallelTensorShape const &rhs_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class ElementBinary : public Op { */
/* public: */
/*   ElementBinary(FFModel &model, */
/*                 OperatorType type, */
/*                 const ParallelTensor x, */
/*                 const ParallelTensor y, */
/*                 bool inplace_a, */
/*                 char const *name); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   void map_output_tensors(FFModel &model) override; */
/*   bool can_inplace_output() override; */
/*   bool has_inplace_output() override; */
/*   void do_inplace_output() override; */
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
/* public: */
/*   bool inplace_a, has_same_operands; */
/*   bool broadcast_input1, broadcast_input2; */
/* }; */

} // namespace FlexFlow

#endif
