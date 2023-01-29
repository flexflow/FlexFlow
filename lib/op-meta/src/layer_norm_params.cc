#include "op-meta/ops/layer_norm_params.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
bool LayerNormParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

typename LayerNormParams::AsConstTuple LayerNormParams::as_tuple() const {
  return {this->axes, this->elementwise_affine, this->eps};
}

bool operator==(LayerNormParams const &lhs, LayerNormParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(LayerNormParams const &lhs, LayerNormParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}
}

namespace std {
size_t hash<FlexFlow::LayerNormParams>::operator()(
    FlexFlow::LayerNormParams const &params) const {
  return get_std_hash(params.as_tuple());
}
}
