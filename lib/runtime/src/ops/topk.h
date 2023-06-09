#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "op-attrs/ops/topk.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<TOPK_INIT_TASK_ID>();
template <>
void register_task<TOPK_FWD_TASK_ID>();
template <>
void register_task<TOPK_BWD_TASK_ID>();

OpTaskInvocation init(TopKAttrs const &);
OpTaskInvocation forward(TopKAttrs const &);
OpTaskInvocation backward(TopKAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  TopKAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class TopK : public Op { */
/* public: */
/*   TopK(FFModel &model, */
/*        ParallelTensor const &input, */
/*        int k, */
/*        bool sorted, */
/*        char const *name); */
/*   TopK(FFModel &model, TopK const &other, ParallelTensor const &input); */
/*   TopK(FFModel &model, */
/*        TopKAttrs const &attrs, */
/*        std::vector<ParallelTensor> const &input, */
/*        char const *name = nullptr); */
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
/*   void serialize(Legion::Serializer &s) const override; */
/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */
/*   Op *materialize(FFModel &ff, */
/*                   ParallelTensor inputs[], */
/*                   int num_inputs) const override; */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */

/* public: */
/*   int k; */
/*   bool sorted; */
/* }; */

} // namespace FlexFlow

#endif
