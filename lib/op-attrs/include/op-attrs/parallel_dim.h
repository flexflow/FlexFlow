#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H

#include "utils/type_traits.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelDim {
  size_t size;
  int degree;
  req<bool> is_replica_dim;
};
FF_VISITABLE_STRUCT(ParallelDim, size, degree, is_replica_dim);
FF_VISIT_FMTABLE(ParallelDim);
CHECK_FMTABLE(ParallelDim);

bool is_valid(ParallelDim const &);
bool is_replica_dim(ParallelDim const &);

} // namespace FlexFlow

#endif
