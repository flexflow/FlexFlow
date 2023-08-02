#ifndef _FLEXFLOW_LIB_PCG_FFI_SRC_PCG_H
#define _FLEXFLOW_LIB_PCG_FFI_SRC_PCG_H

#include "flexflow/pcg.h"
#include "internal/opaque.h"
#include "pcg/computation_graph.h" 
#include "pcg/initializer.h"
#include "pcg/machine_specification.h"
#include "pcg/parallel_computation_graph.h"
#include "pcg/operator.h"
#include "pcg/model_compilation.h"

using namespace FlexFlow;

REGISTER_OPAQUE(flexflow_computation_graph_t, ComputationGraph);
REGISTER_OPAQUE(flexflow_parallel_computation_graph_t, ParallelComputationGraph);
REGISTER_OPAQUE(flexflow_operator_t, Operator);
REGISTER_OPAQUE(flexflow_parallel_tensor_t, ParallelTensor);
REGISTER_OPAQUE(flexflow_layer_t, Layer);
REGISTER_OPAQUE(flexflow_tensor_t, Tensor);
REGISTER_OPAQUE(flexflow_machine_view_t, MachineView);
REGISTER_OPAQUE(flexflow_initializer_t, optional<Initializer>);
REGISTER_OPAQUE(flexflow_machine_specification_t, MachineSpecification);
REGISTER_OPAQUE(flexflow_model_compilation_input_t, ModelCompilationInput);
REGISTER_OPAQUE(flexflow_model_compilation_result_t, ModelCompilationResult);
REGISTER_OPAQUE(flexflow_tensor_list_t, std::vector<Tensor>);
                
struct internal_flexflow_pcg_error_t {
  flexflow_pcg_error_code_t err_code;
};
REGISTER_OPAQUE(flexflow_pcg_error_t, internal_flexflow_pcg_error_t);

#endif
