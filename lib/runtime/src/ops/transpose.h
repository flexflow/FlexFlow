#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "op-attrs/ops/transpose.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<TRANSPOSE_INIT_TASK_ID>();
template <>
void register_task<TRANSPOSE_FWD_TASK_ID>();
template <>
void register_task<TRANSPOSE_BWD_TASK_ID>();

OpTaskInvocation init(TransposeAttrs const &);
OpTaskInvocation forward(TransposeAttrs const &);
OpTaskInvocation backward(TransposeAttrs const &);

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim_factory,
                          TransposeAttrs const &attrs,
                          std::vector<ParallelTensorShape> const &input_shapes,
                          ProfilingSettings const &settings,
                          MachineView const &machine_view);

/* class Transpose : public Op { */
/* public: */
/*   Transpose(FFModel &model, */
/*             TransposeAttrs const &attrs, */
/*             std::vector<ParallelTensor> const &inputs, */
/*             char const *name = nullptr); */
/*   Transpose(FFModel &model, */
/*             ParallelTensor const &input, */
/*             std::vector<int> const &perm, */
/*             char const *name = nullptr); */
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
/*   void init_meta(TransposePerDeviceState *m, */
/*                  Legion::Domain const &in_domain, */
/*                  Legion::Domain const &out_domain) const; */
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
/*   void serialize(Legion::Serializer &s) const override; */
/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */
/*   Op *materialize(FFModel &ff, */
/*                   ParallelTensor inputs[], */
/*                   int num_inputs) const override; */

/* public: */
/*   stack_vector<int, MAX_TENSOR_DIM> perm; */
/* }; */

} // namespace FlexFlow

#endif
