#ifndef _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H
#define _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H

#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "operator.h"
#include "utils/visitable.h"
#include "op-attrs/op.h"

namespace FlexFlow {

struct FusedOpAttrs : public use_visitable_cmp<FusedOpAttrs> {
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

OperatorType get_op_type(FusedOpAttrs const &);

enum SourceType { 
    SOURCE_NONE,
    SOURCE_INPUT, 
    SOURCE_WEIGHT,
     SOURCE_OUTPUT, 
  }; 

struct FusedOP : public Op {
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

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::FusedOpAttrs, graph);
MAKE_VISIT_HASHABLE(::FlexFlow::FusedOpAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<FusedOpAttrs>::value, "");
}

#endif
