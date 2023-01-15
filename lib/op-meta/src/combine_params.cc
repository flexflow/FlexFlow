#include "op-meta/ops/combine_params.h"

namespace FlexFlow {

typename CombineParams::AsConstTuple CombineParams::as_tuple() const {
  return {this->combine_legion_dim, this->combine_degree};
}

bool operator==(CombineParams const &lhs, CombineParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(CombineParams const &lhs, CombineParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool CombineParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &=
      (input.dims[this->combine_legion_dim].degree % this->combine_degree == 0);
  return valid;
}

}
