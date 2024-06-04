#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph.dtg.h"
#include "pcg/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph();

std::unordered_set<parallel_layer_guid_t> get_parallel_layers(ParallelComputationGraph const &);

ParallelTensorAttrs get_parallel_tensor_attrs(ParallelComputationGraph const &, parallel_tensor_guid_t const &);

}

#endif
