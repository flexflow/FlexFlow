#include "op-meta/ops/repartition.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool RepartitionAttrs::is_valid(ParallelTensorShape const &input_shape) const {
  ParallelDim dim = input_shape.at(this->repartition_legion_dim);
  return (dim.size % this->repartition_degree * dim.degree == 0);
}

bool operator==(RepartitionAttrs const &lhs, RepartitionAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(RepartitionAttrs const &lhs, RepartitionAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::RepartitionAttrs;

size_t hash<RepartitionAttrs>::operator()(
    RepartitionAttrs const &params) const {
  return visit_hash(params);
}
}
