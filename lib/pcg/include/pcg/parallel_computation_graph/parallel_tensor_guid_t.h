#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_TENSOR_GUID_T_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_TENSOR_GUID_T_H

#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

parallel_layer_guid_t get_source_layer(parallel_tensor_guid_t const &);

} // namespace FlexFlow

#endif
