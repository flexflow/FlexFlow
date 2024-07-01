#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

namespace FlexFlow {
struct Config {
  Config(void);
  int batchSize;
};

class ResNet {
  public:
  ComputationGraph create_computation_graph(Config &config);
  ParallelComputationGraph create_parallel_computation_graph(Config &config);
};

} // namespace FlexFlow
