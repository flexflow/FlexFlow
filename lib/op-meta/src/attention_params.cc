#include "op-meta/ops/attention_params.h"
#include "utils/hash-utils.h"
#include <algorithm>
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool MultiHeadAttentionParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(), [](ParallelTensorShape const &s) { return s.is_valid(); }));
  bool is_valid = true;
  return is_valid;
}

bool operator==(MultiHeadAttentionParams const &lhs, MultiHeadAttentionParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(MultiHeadAttentionParams const &lhs, MultiHeadAttentionParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::MultiHeadAttentionParams;

size_t hash<MultiHeadAttentionParams>::operator()(
    MultiHeadAttentionParams const &params) const {
  return visit_hash(params);
} 
}
