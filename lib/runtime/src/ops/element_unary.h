#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "op-attrs/ops/element_unary.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<ELEMENTUNARY_INIT_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_FWD_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_BWD_TASK_ID>();

OpTaskInvocation init(ElementUnaryAttrs const &);
OpTaskInvocation forward(ElementUnaryAttrs const &);
OpTaskInvocation backward(ElementUnaryAttrs const &);

OpTaskInvocation init(ElementScalarUnaryAttrs const &);
OpTaskInvocation forward(ElementScalarUnaryAttrs const &);
OpTaskInvocation backward(ElementScalarUnaryAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementUnaryAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementScalarUnaryAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class ElementUnary : public Op { */
/* public: */
/*   ElementUnary(FFModel &model, */
/*                OperatorType type, */
/*                const ParallelTensor x, */
/*                bool inplace, */
/*                char const *name, */
/*                float scalar); */
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
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */

/* private: */
/*   bool inplace; */

/* public: */
/*   float scalar; */
/* }; */

} // namespace FlexFlow

#endif
