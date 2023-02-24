#include "op-meta/ops/layer_norm_params.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(LayerNormParams const &lhs, LayerNormParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(LayerNormParams const &lhs, LayerNormParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::LayerNormParams;

size_t hash<LayerNormParams>::operator()(
    LayerNormParams const &params) const {
  return visit_hash(params);
}
}
