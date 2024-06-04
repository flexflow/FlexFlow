#include "op-attrs/shard_parallel_dim.h"

namespace FlexFlow {

bool is_valid(ShardParallelDim const &d) {
  return d.degree > 0 && d.size > 0 && (d.size % d.degree) == 0;
}

} // namespace FlexFlow
