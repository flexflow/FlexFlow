#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.dtg.h"
#include "pcg/layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "utils/json.h"

namespace FlexFlow {

using V1ComputationGraph = V1LabelledDataflowGraph<LayerAttrs, TensorAttrs>;
CHECK_IS_JSONABLE(V1ComputationGraph);
V1ComputationGraph to_v1(ComputationGraph const &);

using V1ParallelComputationGraph =
    V1LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>;
CHECK_IS_JSONABLE(V1ParallelComputationGraph);
V1ParallelComputationGraph to_v1(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
