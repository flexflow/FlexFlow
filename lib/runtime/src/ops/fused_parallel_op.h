#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_H

#include "fused_parallel_op_attrs.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<FUSED_PARALLELOP_INIT_TASK_ID>();
template <>
void register_task<FUSED_PARALLELOP_FWD_TASK_ID>();
template <>
void register_task<FUSED_PARALLELOP_BWD_TASK_ID>();

OpTaskInvocation init(FusedParallelOpAttrs const &);
OpTaskInvocation forward(FusedParallelOpAttrs const &);
OpTaskInvocation backward(FusedParallelOpAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &factory,
                                  FusedParallelOpAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class FusedParallelOp : public ParallelOp { */
/* public: */
/*   FusedParallelOp(FFModel &model, */
/*                   const ParallelTensor input, */
/*                   std::vector<ParallelOpInfo> const &parallel_ops); */
/*   FusedParallelOp(FFModel &model, FusedParallelOpAttrs const &attrs,
 * std::vector<ParallelTensor> const &inputs); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   bool append_parallel_op_info( */
/*       std::vector<ParallelOpInfo> &parallel_ops) const override; */
/*   void create_input_partition(FFModel &model) override; */
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
/*       forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);
 */
/*   template <typename T> */
/*   static void backward_kernel(T const *output_grad_ptr, */
/*                               T *input_grad_ptr, */
/*                               size_t num_elements); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &mv, */
/*                              CostMetrics &cost_metrics) const override; */
/*   void set_parallel_ops(std::vector<ParallelOpInfo> const &_parallel_ops); */
/*   bool check_no_redundant_parallel_ops(void) const; */
/* public: */
/*   stack_vector<ParallelOpInfo, MAX_NUM_FUSED_OPERATORS> parallel_ops; */
/* }; */

} // namespace FlexFlow

#endif
