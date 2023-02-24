#include "op-meta/ops/repartition_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool RepartitionParams::is_valid(ParallelTensorShape const &input_shape) const {
  ParallelDim dim = input_shape.at(this->repartition_legion_dim);
  return (dim.size % this->repartition_degree * dim.degree == 0);
}

bool operator==(RepartitionParams const &lhs, RepartitionParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(RepartitionParams const &lhs, RepartitionParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::RepartitionParams;

size_t hash<RepartitionParams>::operator()(
    RepartitionParams const &params) const {
  return visit_hash(params);
}
}
