#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MODEL_COMPILATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MODEL_COMPILATION_H

#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/tensor_mapping.h"

namespace FlexFlow {

struct ModelCompilationInput {
  ComputationGraph computation_graph;
  Optimizer optimizer;
};
FF_VISITABLE_STRUCT(ModelCompilationInput, computation_graph, optimizer);

struct ModelCompilationResult {
  ModelCompilationInput input;
  ParallelComputationGraph pcg;
  req<TensorMapping> tensor_mapping;
};
FF_VISITABLE_STRUCT(ModelCompilationResult, input, pcg, tensor_mapping);

} // namespace FlexFlow

#endif
