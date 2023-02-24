#include "op-meta/ops/split_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool SplitParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return inputs.size() == 1 && inputs.at(0).is_valid();
}

int SplitParams::num_outputs(std::vector<ParallelTensorShape> const &inputs) const {
  return splits.size();
}

OperatorType SplitParams::op_type() const {
  return OP_SPLIT;
}

bool operator==(SplitParams const &lhs, SplitParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(SplitParams const &lhs, SplitParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::SplitParams;

size_t hash<SplitParams>::operator()(SplitParams const &p) const {
  return visit_hash(p);
}
}
