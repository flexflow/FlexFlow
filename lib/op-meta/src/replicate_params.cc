#include "op-meta/ops/replicate_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool ReplicateParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return inputs.size() == 1 && inputs.at(0).is_valid();
}

typename ReplicateParams::AsConstTuple ReplicateParams::as_tuple() const {
  return {this->replicate_legion_dim, this->replicate_degree};
}

bool operator==(ReplicateParams const &lhs, ReplicateParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ReplicateParams const &lhs, ReplicateParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {
size_t hash<FlexFlow::ReplicateParams>::operator()(
    FlexFlow::ReplicateParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
