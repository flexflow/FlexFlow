#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REPLICA_PARALLEL_DIM_SET_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REPLICA_PARALLEL_DIM_SET_H

#include "op-attrs/replica_parallel_dim.dtg.h"
#include "op-attrs/replica_parallel_dim_set.dtg.h"
#include "op-attrs/replica_type.dtg.h"

namespace FlexFlow {

ReplicaParallelDimSet empty_replica_parallel_dim_set();
int get_degree_of_replica_type(ReplicaParallelDimSet const &, ReplicaType);
std::unordered_set<ReplicaParallelDim> get_replica_dims(ReplicaParallelDimSet const &);
bool is_valid(ReplicaParallelDimSet const &);

} // namespace FlexFlow

#endif
