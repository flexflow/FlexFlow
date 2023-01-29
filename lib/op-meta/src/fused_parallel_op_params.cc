#include "op-meta/ops/fused_parallel_op_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool FusedParallelOpParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

typename FusedParallelOpParams::AsConstTuple FusedParallelOpParams::as_tuple() const {
  return {this->parallel_ops};
}

bool operator==(FusedParallelOpParams const &lhs, FusedParallelOpParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(FusedParallelOpParams const &lhs, FusedParallelOpParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::FusedParallelOpParams>::operator()(
    FlexFlow::FusedParallelOpParams const &p) const {
  return get_std_hash(p.as_tuple());
}
}
