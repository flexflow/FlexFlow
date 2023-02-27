#include "op-meta/ops/gather_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool GatherParams::is_valid(ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
  if (lhs.num_dims() != rhs.num_dims()) {
    return false;
  }
  for (int i = 0; i < lhs.num_dims(); i++) {
    if (i != this->legion_dim &&
        lhs.at(i).size < rhs.at(i).size) {
      return false;
    }
  }
  return true;
}

bool operator==(GatherParams const &lhs, GatherParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(GatherParams const &lhs, GatherParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::GatherParams;

size_t hash<GatherParams>::operator()(GatherParams const &p) const {
  return visit_hash(p);
}

}
