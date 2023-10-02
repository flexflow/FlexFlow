#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_DIM_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_PARALLEL_DIM_H

#include "op-attrs/parallel_dim.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ParallelDim {
  size_t size;
  int degree;
  req<bool> is_replica_dim;
};
FF_VISITABLE_STRUCT(V1ParallelDim, size, degree, is_replica_dim);
CHECK_IS_JSONABLE(V1ParallelDim);

V1ParallelDim to_v1(ParallelDim const &dim);

} // namespace FlexFlow

#endif
