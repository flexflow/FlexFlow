#include "pcg/file_format/v1/ops/layer_norm.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1LayerNormAttrs to_v1(LayerNormAttrs const &a) {
  return {std::vector<int>(a.axes.begin(), a.axes.end()),
          to_v1(a.elementwise_affine),
          to_v1(a.eps)};
}

} // namespace FlexFlow
