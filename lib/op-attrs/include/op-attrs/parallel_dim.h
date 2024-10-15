#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H

#include "op-attrs/parallel_dim.dtg.h"

namespace FlexFlow {

bool is_valid(ParallelDim const &);
bool is_replica_dim(ParallelDim const &);

ParallelDim with_size_set_to(ParallelDim const &, size_t);
ParallelDim with_degree_set_to(ParallelDim const &, int);
ParallelDim with_is_replica_set_to(ParallelDim const &, bool);
int get_degree(ParallelDim const &);

} // namespace FlexFlow

#endif
