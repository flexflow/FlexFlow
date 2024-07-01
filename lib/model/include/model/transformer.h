#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

namespace FlexFlow {

class Transformer {
  public:
  ComputationGraph create_computation_graph(Config &config);
  ParallelComputationGraph create_parallel_computation_graph();
}


struct Config {
  Config(void);
  int hidden_size, embedding_size, num_heads, num_layers, sequence_length,
      batchSize;
};

} // namespace FlexFlow
