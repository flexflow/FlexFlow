#ifndef _FLEXFLOW_LIB_PCG_FFI_SRC_PCG_H
#define _FLEXFLOW_LIB_PCG_FFI_SRC_PCG_H

#include "flexflow/pcg.h"
#include "internal/opaque.h"
#include "pcg/computation_graph.h"
#include "pcg/initializer.h"
#include "pcg/layer.h"
#include "pcg/layer_guid_t.h"
#include "pcg/machine_specification.h"
#include "pcg/model_compilation.h"
#include "pcg/operator.h"
#include "pcg/operator_guid_t.h"
#include "pcg/parallel_computation_graph.h"
#include "pcg/parallel_tensor.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "pcg/tensor_guid_t.h"
#include "utils/type_traits.h"

using namespace FlexFlow;

namespace FlexFlow {

struct internal_flexflow_layer_t {
  layer_guid_t guid;
  Layer layer;
};
FF_VISITABLE_STRUCT(internal_flexflow_layer_t, guid, layer);

struct internal_flexflow_tensor_t {
  tensor_guid_t guid;
  Tensor tensor;
};
FF_VISITABLE_STRUCT(internal_flexflow_tensor_t, guid, tensor);

struct internal_flexflow_operator_t {
  operator_guid_t guid;
  Operator op;
};
FF_VISITABLE_STRUCT(internal_flexflow_operator_t, guid, op);

struct internal_flexflow_parallel_tensor_t {
  parallel_tensor_guid_t guid;
  ParallelTensor parallel_tensor;
};
FF_VISITABLE_STRUCT(internal_flexflow_parallel_tensor_t, guid, parallel_tensor);

} // namespace FlexFlow

REGISTER_OPAQUE(flexflow_computation_graph_t, ComputationGraph);
REGISTER_OPAQUE(flexflow_parallel_computation_graph_t,
                ParallelComputationGraph);
REGISTER_OPAQUE(flexflow_operator_t, internal_flexflow_operator_t);
REGISTER_OPAQUE(flexflow_parallel_tensor_t,
                internal_flexflow_parallel_tensor_t);
REGISTER_OPAQUE(flexflow_layer_t, internal_flexflow_layer_t);
REGISTER_OPAQUE(flexflow_tensor_t, internal_flexflow_tensor_t);
REGISTER_OPAQUE(flexflow_machine_view_t, MachineView);
REGISTER_OPAQUE(flexflow_initializer_t, optional<Initializer>);
REGISTER_OPAQUE(flexflow_machine_specification_t, MachineSpecification);
REGISTER_OPAQUE(flexflow_model_compilation_input_t, ModelCompilationInput);
REGISTER_OPAQUE(flexflow_model_compilation_result_t, ModelCompilationResult);
REGISTER_OPAQUE(flexflow_tensor_list_t,
                std::vector<internal_flexflow_tensor_t>);

struct internal_flexflow_pcg_error_t {
  flexflow_pcg_error_code_t err_code;
};
REGISTER_OPAQUE(flexflow_pcg_error_t, internal_flexflow_pcg_error_t);

#endif
