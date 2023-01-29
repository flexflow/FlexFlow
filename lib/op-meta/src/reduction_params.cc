#include "op-meta/ops/reduction_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

typename ReductionParams::AsConstTuple ReductionParams::as_tuple() const {
  return {this->reduction_legion_dim, this->reduction_degree};
}

bool operator==(ReductionParams const &lhs, ReductionParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ReductionParams const &lhs, ReductionParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool ReductionParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

}

namespace std {
size_t hash<FlexFlow::ReductionParams>::operator()(
    FlexFlow::ReductionParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
