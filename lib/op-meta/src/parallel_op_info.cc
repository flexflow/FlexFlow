#include "op-meta/parallel_op_info.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

typename ParallelOpInfo::AsConstTuple ParallelOpInfo::as_tuple() const {
  return {this->op_type, this->parallel_dim, this->parallel_degree};
}

bool operator==(ParallelOpInfo const &lhs, ParallelOpInfo const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ParallelOpInfo const &lhs, ParallelOpInfo const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::ParallelOpInfo>::operator()(
    FlexFlow::ParallelOpInfo const &params) const {
  return get_std_hash(params.as_tuple());
}
}
