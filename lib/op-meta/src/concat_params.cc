#include "op-meta/ops/concat_params.h"

namespace FlexFlow {

typename ConcatParams::AsConstTuple ConcatParams::as_tuple() const {
  return {this->axis};
}

bool operator==(ConcatParams const &lhs, ConcatParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ConcatParams const &lhs, ConcatParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool ConcatParams::is_valid(
    std::vector<ParallelTensorShape> const &input) const {
  bool valid = true;
  for (auto p : input) {
    valid &= p.is_valid();
  }
  return valid;
}

}
