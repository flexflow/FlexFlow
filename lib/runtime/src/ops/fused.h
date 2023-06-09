#ifndef _FLEXFLOW_FUSED_H_
#define _FLEXFLOW_FUSED_H_

#include "fused_op_attrs.h"
#include "op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<FUSEDOP_INIT_TASK_ID>();
template <>
void register_task<FUSEDOP_FWD_TASK_ID>();
template <>
void register_task<FUSEDOP_BWD_TASK_ID>();

OpTaskInvocation init(FusedOpAttrs const &);
OpTaskInvocation forward(FusedOpAttrs const &);
OpTaskInvocation backward(FusedOpAttrs const &);

/* class FusedPerDeviceOpState : public PerDeviceOpState { */
/* public: */
/*   FusedPerDeviceOpState(void) {} */
/*   PerDeviceOpState *meta[MAX_NUM_FUSED_OPERATORS]; */
/*   FusedOp *fused_op; */
/*   int numOperators; */
/* }; */

/* class FusedOp : public Op { */
/* public: */
/*   enum SourceType { */
/*     SOURCE_NONE, */
/*     SOURCE_INPUT, */
/*     SOURCE_WEIGHT, */
/*     SOURCE_OUTPUT, */
/*   }; */
/*   FusedOp(FFModel &model, Op *op); */
/*   bool add_operator(FFModel &model, Op *op); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
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
/*   TaskID get_init_task_id() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   TaskID get_fwd_task_id() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */
/*   TaskID get_bwd_task_id() const override; */

/* public: */
/*   FFIterationConfig iter_config; */
/*   int op_num_inputs[MAX_NUM_FUSED_OPERATORS]; */
/*   int op_num_weights[MAX_NUM_FUSED_OPERATORS]; */
/*   int op_num_outputs[MAX_NUM_FUSED_OPERATORS]; */
/*   OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS]; */
/*   SourceType op_input_source[MAX_NUM_FUSED_TENSORS]; */
/*   SourceType op_weight_source[MAX_NUM_FUSED_TENSORS]; */
/*   SourceType op_output_source[MAX_NUM_FUSED_TENSORS]; */
/*   DataType input_data_types[MAX_NUM_INPUTS]; */
/*   DataType weight_data_types[MAX_NUM_WEIGHTS]; */
/*   DataType output_data_types[MAX_NUM_OUTPUTS]; */
/*   int op_input_idx[MAX_NUM_FUSED_TENSORS]; */
/*   int op_weight_idx[MAX_NUM_FUSED_TENSORS]; */
/*   int op_output_idx[MAX_NUM_FUSED_TENSORS]; */
/*   Op *operators[MAX_NUM_FUSED_OPERATORS]; */
/*   FusedPerDeviceOpState fused_meta[MAX_NUM_WORKERS]; */
/*   int numOperators; */
/* }; */

} // namespace FlexFlow

#endif
