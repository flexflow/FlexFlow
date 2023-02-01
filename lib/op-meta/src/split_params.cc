#include "op-meta/ops/split_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool SplitParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return inputs.size() == 1 && inputs.at(0).is_valid();
}

typename SplitParams::AsConstTuple SplitParams::as_tuple() const {
  return {this->splits, this->legion_axis};
}

int SplitParams::num_outputs(std::vector<ParallelTensorShape> const &inputs) const {
  return splits.size();
}

OperatorType SplitParams::op_type() const {
  return OP_SPLIT;
}

bool operator==(SplitParams const &lhs, SplitParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(SplitParams const &lhs, SplitParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::SplitParams>::operator()(FlexFlow::SplitParams const &p) const {
  return get_std_hash(p);
}
}
