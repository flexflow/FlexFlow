#ifndef _FLEXFLOW_COMPILER_GRAPH_UTILS_H
#define _FLEXFLOW_COMPILER_GRAPH_UTILS_H

#include "compiler/unity_algorithm.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

SeriesParallelDecomposition
    get_series_parallel_decomposition(ParallelComputationGraph const &pcg);

ParallelComputationGraph cg_to_pcg(ComputationGraph const &g);
SubParallelComputationGraph pcg_to_subpcg(ParallelComputationGraph const &g);

// NOTE(@wmdi): I think we should have the following interfaces in the graph
// library eventually.

template <typename T>
void minimize(T &t, T const &v) {
  if (v < t) {
    t = v;
  }
}

template <typename T, typename Compare>
void minimize(T &t, T const &v, Compare comp) {
  if (comp(v, t)) {
    t = v;
  }
}

} // namespace FlexFlow

#endif
