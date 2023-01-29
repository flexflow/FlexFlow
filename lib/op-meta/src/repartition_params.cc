#include "op-meta/ops/repartition_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool RepartitionParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &= (input.at(this->repartition_legion_dim).size %
                (this->repartition_degree *
                 input.at(this->repartition_legion_dim).degree) ==
            0);
  return valid;
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
