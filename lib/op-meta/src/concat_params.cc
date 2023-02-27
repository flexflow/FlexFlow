#include "op-meta/ops/concat_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(ConcatParams const &lhs, ConcatParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ConcatParams const &lhs, ConcatParams const &rhs) {
  return visit_lt(lhs, rhs);
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
}

namespace std {
using ::FlexFlow::opmeta::ConcatParams;

size_t hash<ConcatParams>::operator()(ConcatParams const &p) const {
  return visit_hash(p);
}

}
