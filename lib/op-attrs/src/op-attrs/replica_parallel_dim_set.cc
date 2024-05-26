#include "op-attrs/replica_parallel_dim_set.h"
#include "utils/exception.h"

namespace FlexFlow {

ReplicaParallelDimSet empty_replica_parallel_dim_set() {
  return ReplicaParallelDimSet{1, 1};
}

int get_order_of_replica_type(ReplicaParallelDimSet const &s,
                              ReplicaType replica_type) {
  switch (replica_type) {
    case ReplicaType::SUM:
      return s.sum_degree;
    case ReplicaType::DISCARD_COPY:
      return s.discard_copy_degree;
    default:
      throw mk_runtime_error(fmt::format("Unexpected ReplicaType value: {}",
                                         static_cast<int>(replica_type)));
  }
}

std::unordered_set<ReplicaParallelDim>
    get_replica_dims(ReplicaParallelDimSet const &s) {
  return std::unordered_set<ReplicaParallelDim>{
      ReplicaParallelDim{s.sum_degree, ReplicaType::SUM},
      ReplicaParallelDim{s.discard_copy_degree, ReplicaType::DISCARD_COPY},
  };
}

bool is_valid(ReplicaParallelDimSet const &s) {
  return s.sum_degree > 0 && s.discard_copy_degree > 0;
}

} // namespace FlexFlow
