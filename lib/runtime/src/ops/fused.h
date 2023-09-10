#ifndef _FLEXFLOW_FUSED_H_
#define _FLEXFLOW_FUSED_H_

#include "fused_op_attrs.h"
#include "op_task_invocation.h"
#include "utils/variant.h"
#include "kernel/batch_matmul_kernels.h"
#include "kernels/batch_norm_kernels.h"
#include "kernel/reduce_kernels.h"
#include "kernel/cast_kernels.h"
#include "kernel/combine_kernels.h"
#include "kernel/concat_kernels.h"
#include "kernel/conv_2d_kernels.h"
#include "kernel/dropout_kernels.h"
#include "kernel/element_binary_kernels.h"
#include "kernel/element_unary_kernels.h"
#include "kernel/embedding_kernels.h"
#include "kernel/flat_kernels.h"
#include "kernel/gather_kernels.h"
#include "kernel/groupby_kernels.h"
#include "kernel/layer_norm_kernels.h"
#include "kernel/linear_kernels.h"
#include "kernel/partition_kernels.h"
#include "kernel/pool_2d_kernels.h"
#include "kernel/reduce_kernels.h"
#include "kernel/reshape_kernels.h"
#include "kernel/softmax_kernels.h"
#include "kernel/top_k_kernels.h"
#include "kernel/transpose_kernels.h"
#include "config.h"

namespace FlexFlow {

enum SourceType { 
    SOURCE_NONE,
    SOURCE_INPUT, 
    SOURCE_WEIGHT,
     SOURCE_OUTPUT, 
  }; 

class FusedOP;

using AllDevice = variant<BatchMatmulPerDeviceState, BatchNormPerDeviceState, CastPerDeviceState, CombinePerDeviceState, ConcatPerDeviceState,  Conv2DPerDeviceState, DropoutPerDeviceState, ElementBinaryPerDeviceState, ElementUnaryPerDeviceState, EmbeddingPerDeviceState, FlatPerDeviceState, GatherPerDeviceState, GroupByPerDeviceState, LayerNormPerDeviceState, LinearPerDeviceState, RepartitionPerDeviceState, Pool2DPerDeviceState, ReducePerDeviceState, ReshapePerDeviceState, SoftmaxPerDeviceState, TopKPerDeviceState, TransposePerDeviceState>;//this can update to hold all device state

struct FusedPerDeviceOpState{
    FusedOp fused_op;
    int numOperators;
    //maybe we can all xxxPerdeviceOpState here，use Variant to store all device 

};

struct FusedOP : public OP {
    int numOperators; 
    int numInputs;
    int numWeights;
    int numOutputs;
    
    FFIterationConfig iter_config; 
    int op_num_inputs[MAX_NUM_FUSED_OPERATORS]; 
    int op_num_weights[MAX_NUM_FUSED_OPERATORS]; 
    int op_num_outputs[MAX_NUM_FUSED_OPERATORS]; 
    OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS]; 
    SourceType op_input_source[MAX_NUM_FUSED_TENSORS]; 
    SourceType op_weight_source[MAX_NUM_FUSED_TENSORS]; 
    SourceType op_output_source[MAX_NUM_FUSED_TENSORS];
    DataType input_data_types[MAX_NUM_INPUTS]; 
    DataType weight_data_types[MAX_NUM_WEIGHTS]; 
    DataType output_data_types[MAX_NUM_OUTPUTS];
    int op_input_idx[MAX_NUM_FUSED_TENSORS]; */
    int op_weight_idx[MAX_NUM_FUSED_TENSORS]; */
    int op_output_idx[MAX_NUM_FUSED_TENSORS]; */
    Op *operators[MAX_NUM_FUSED_OPERATORS]; */
    FusedPerDeviceOpState fused_meta[MAX_NUM_WORKERS]; 
    int numOperators; 

};

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
