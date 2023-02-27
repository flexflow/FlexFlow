#include "op-meta/ops/batch_matmul_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(BatchMatmulParams const &lhs, BatchMatmulParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(BatchMatmulParams const &lhs, BatchMatmulParams const &rhs) {
  return visit_lt(lhs, rhs);
}

bool BatchMatmulParams::is_valid(
    ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
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
}

namespace std {

using ::FlexFlow::opmeta::BatchMatmulParams;

size_t hash<BatchMatmulParams>::operator()(BatchMatmulParams const &p) const {
  return visit_hash(p);
}

}
