#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_EDGE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_EDGE_H

#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_use_t.dtg.h"
#include "substitutions/open_parallel_tensor_guid_t.dtg.h"
#include "substitutions/sub_parallel_computation_graph_edge.dtg.h"

namespace FlexFlow {

SubParallelComputationGraphEdge
    subpcg_edge_from_tensor_and_dst(parallel_tensor_guid_t const &tensor,
                                    parallel_layer_guid_t const &layer,
                                    int input_idx);
SubParallelComputationGraphEdge
    subpcg_edge_from_tensor_and_use(open_parallel_tensor_guid_t const &tensor,
                                    parallel_tensor_use_t const &use);
open_parallel_tensor_guid_t
    get_parallel_tensor(SubParallelComputationGraphEdge const &);

} // namespace FlexFlow

#endif
