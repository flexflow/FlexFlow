#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_SERIALIZATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_SERIALIZATION_H

#include "computation_graph.h"
#include "layer.h"
#include "machine_specification.h"
#include "parallel_computation_graph.h"
#include "parallel_tensor.h"
#include "tensor_mapping.h"
#include "utils/json.h"

namespace FlexFlow {

void from_json(json const &, ComputationGraph &);
void to_json(json &, ComputationGraph const &);

void from_json(json const &, ParallelComputationGraph &);
void to_json(json &, ParallelComputationGraph const &);

void from_json(json const &, Layer &);
void to_json(json &, Layer const &);

void from_json(json const &, ParallelTensor &);
void to_json(json &, ParallelTensor const &);

void from_json(json const &, Tensor &);
void to_json(json &, Tensor const &);

void from_json(json const &, Initializer &);
void to_json(json &, Initializer const &);

void from_json(json const &, MachineSpecification &);
void to_json(json &, MachineSpecification const &);

void from_json(json const &, Operator &);
void to_json(json &, Operator const &);

void from_json(json const &, MachineView &);
void to_json(json &, MachineView const &);

void from_json(json const &, StridedRectangle &);
void to_json(json &, StridedRectangle const &);

void from_json(json const &, StridedRectangleSide &);
void to_json(json &, StridedRectangleSide const &);

void from_json(json const &, ParallelTensorDims &);
void to_json(json &, ParallelTensorDims const &);

void from_json(json const &, TensorDims &);
void to_json(json &, TensorDims const &);

void from_json(json const &, TensorMapping &);
void to_json(json &, TensorMapping const &);

void from_json(json const &, ParallelTensorShape &);
void to_json(json &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
