#include "op-attrs/parallel_dim.h"

namespace FlexFlow {

bool is_valid(ParallelDim const &dim) {
  return dim.size > 0 && dim.degree >= 1 && dim.size % dim.degree == 0;
}

bool is_replica_dim(ParallelDim const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
