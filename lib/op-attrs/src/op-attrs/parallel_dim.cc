#include "op-attrs/parallel_dim.h"
#include "utils/overload.h"

namespace FlexFlow {

int get_degree(ParallelDim const &dim) {
  return dim.visit<int>(overload{
      [](ShardParallelDim const &shard_dim) { return shard_dim.degree; },
      [](ReplicaParallelDim const &replica_dim) {
        return replica_dim.degree;
      }});
}

} // namespace FlexFlow
