#include "op-meta/ops/batch_matmul_params.h"

namespace FlexFlow {

typename BatchMatmulParams::AsConstTuple BatchMatmulParams::as_tuple() const {
  return {this->a_seq_length_dim, this->b_seq_length_dim};
}

bool operator==(BatchMatmulParams const &lhs, BatchMatmulParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(BatchMatmulParams const &lhs, BatchMatmulParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool BatchMatmulParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  if (!input.first.is_valid()) {
    return false;
  }
  if (!input.second.is_valid()) {
    return false;
  }
  if (input.first.num_dims != input.second.num_dims) {
    return false;
  }
  for (int i = input.first.num_dims - 1; i >= 2; i--) {
    if (input.first.dims[i] != input.second.dims[i]) {
      return false;
    }
  }
  if (input.first.dims[0] != input.second.dims[1]) {
    return false;
  }
  return true;
}

}
