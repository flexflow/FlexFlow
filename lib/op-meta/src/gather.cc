#include "op-meta/ops/gather.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool GatherAttrs::is_valid(ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
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

bool operator==(GatherAttrs const &lhs, GatherAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(GatherAttrs const &lhs, GatherAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::GatherAttrs;

size_t hash<GatherAttrs>::operator()(GatherAttrs const &p) const {
  return visit_hash(p);
}

}
