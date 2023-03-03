#include "op-meta/ops/layer_norm.h"
#include "utils/hash-utils.h"
#include "utils/visit_struct.h"

namespace FlexFlow {

bool operator==(LayerNormAttrs const &lhs, LayerNormAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(LayerNormAttrs const &lhs, LayerNormAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::LayerNormAttrs;

size_t hash<LayerNormAttrs>::operator()(
    LayerNormAttrs const &params) const {
  return visit_hash(params);
}
}
