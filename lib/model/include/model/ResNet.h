#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {
struct Config {
  Config(void);
  int batchSize;
};
ComputationGraph create_computation_graph();
ParallelComputationGraph create_parallel_computation_graph();
} // namespace FlexFlow