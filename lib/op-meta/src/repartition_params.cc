#include "op-meta/ops/repartition_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool RepartitionParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  if (inputs.size() != 1 || !inputs.at(0).is_valid()) { 
    return false; 
  }

  ParallelDim dim = inputs.at(0).at(this->repartition_legion_dim);
  return (dim.size % this->repartition_degree * dim.degree == 0);
}

typename RepartitionParams::AsConstTuple RepartitionParams::as_tuple() const {
  return {this->repartition_legion_dim, this->repartition_degree};
}

bool operator==(RepartitionParams const &lhs, RepartitionParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(RepartitionParams const &lhs, RepartitionParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::RepartitionParams>::operator()(
    FlexFlow::RepartitionParams const &params) const {
  return get_std_hash(params.as_tuple());
}
}
