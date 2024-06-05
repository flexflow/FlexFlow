#include "op-attrs/replica_parallel_dim.h"

namespace FlexFlow {

bool is_valid(ReplicaParallelDim const &d) {
  return d.degree > 0;
}

} // namespace FlexFlow
