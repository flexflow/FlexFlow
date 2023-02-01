#include "op-meta/ops/attention_params.h"
#include "utils/hash-utils.h"
#include <algorithm>

namespace FlexFlow {

bool MultiHeadAttentionParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(), [](ParallelTensorShape const &s) { return s.is_valid(); }));
  bool is_valid = true;
  return is_valid;
}

typename MultiHeadAttentionParams::AsConstTuple MultiHeadAttentionParams::as_tuple() const {
  return {this->embed_dim, this->num_heads, this->kdim, this->vdim, this->dropout, this->bias, this->add_bias_kv, this->add_zero_attn};
}

bool operator==(MultiHeadAttentionParams const &lhs, MultiHeadAttentionParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(MultiHeadAttentionParams const &lhs, MultiHeadAttentionParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}
}

namespace std {
size_t hash<FlexFlow::MultiHeadAttentionParams>::operator()(
    FlexFlow::MultiHeadAttentionParams const &params) const {
  return get_std_hash(params.as_tuple());
} 
}
