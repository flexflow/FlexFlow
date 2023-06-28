#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H

#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelDim {
  req<size_t> size;
  req<int> degree;
  req<bool> is_replica_dim;
};
FF_VISITABLE_STRUCT(ParallelDim, size, degree, is_replica_dim);

bool is_valid(ParallelDim const &);
bool is_replica_dim(ParallelDim const &);

} // namespace FlexFlow

#endif
