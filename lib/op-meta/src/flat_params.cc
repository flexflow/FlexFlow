#include "op-meta/ops/flat_params.h"

namespace FlexFlow {

typename FlatParams::AsConstTuple FlatParams::as_tuple() const {
  return {};
}

bool operator==(FlatParams const &lhs, FlatParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(FlatParams const &lhs, FlatParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool FlatParams::is_valid(ParallelTensorShape const &input) const {
  ParallelTensorShape output_shape;

  this->solve_dims(input, output_shape);

  bool is_valid = true;
  is_valid &= input.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= (input.dims[FlatInput::WIDTH].degree == 1);

  return is_valid;
}


}
