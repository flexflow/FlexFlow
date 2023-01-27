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
  ParallelTensorShape lhs = input.first;
  ParallelTensorShape rhs = input.second;

  if (!lhs.is_valid() || !rhs.is_valid()) {
    return false;
  }
  if (lhs.num_dims() != rhs.num_dims()) {
    return false;
  }
  for (int i = lhs.num_dims() - 1; i >= 2; i--) {
    if (lhs.at(i) != rhs.at(i)) {
      return false;
    }
  }
  if (lhs.at(0) != rhs.at(1)) {
    return false;
  }

  return true;
}

}
