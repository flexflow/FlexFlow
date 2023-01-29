#include "op-meta/ops/dropout_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool DropoutParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

typename DropoutParams::AsConstTuple DropoutParams::as_tuple() const {
  return {this->rate, this->seed};
}

bool operator==(DropoutParams const &lhs, DropoutParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(DropoutParams const &lhs, DropoutParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}
}

namespace std {
size_t hash<FlexFlow::DropoutParams>::operator()(
    FlexFlow::DropoutParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
