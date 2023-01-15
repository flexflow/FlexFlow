#include "op-meta/ops/cast_params.h"

namespace FlexFlow {

typename CastParams::AsConstTuple CastParams::as_tuple() const {
  return {this->dtype};
}

bool operator==(CastParams const &lhs, CastParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(CastParams const &lhs, CastParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool CastParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &= (input.dims[input.num_dims - 1].degree == 1);
  return valid;
}

}
