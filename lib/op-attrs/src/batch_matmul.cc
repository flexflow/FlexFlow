#include "op-attrs/ops/batch_matmul.h"
#include "utils/visit_struct.h"

namespace FlexFlow {

bool operator==(BatchMatmulAttrs const &lhs, BatchMatmulAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(BatchMatmulAttrs const &lhs, BatchMatmulAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

bool BatchMatmulAttrs::is_valid(
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

namespace std {

using ::FlexFlow::BatchMatmulAttrs;

size_t hash<BatchMatmulAttrs>::operator()(BatchMatmulAttrs const &p) const {
  return visit_hash(p);
}

}
