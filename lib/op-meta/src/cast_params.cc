#include "op-meta/ops/cast_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(CastParams const &lhs, CastParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(CastParams const &lhs, CastParams const &rhs) {
  return visit_lt(lhs, rhs);
}

bool CastParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &= (input.at(input.num_dims() - 1).degree == 1);
  return valid;
}

}
}

namespace std {
using ::FlexFlow::opmeta::CastParams;

size_t hash<CastParams>::operator()(
    CastParams const &params) const {
  return visit_hash(params);
} 
}

