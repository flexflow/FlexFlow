#include "pcg/file_format/v1/ops/layer_norm.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1LayerNormAttrs to_v1(LayerNormAttrs const &a) {
  return {std::vector<int>(a.axes.begin(), a.axes.end()),
          a.elementwise_affine,
          a.eps};
}

LayerNormAttrs from_v1(V1LayerNormAttrs const &va) {

  return {
      stack_vector<ff_dim_t, MAX_TENSOR_DIM>(va.axes.begin(), va.axes.end()),
      va.elementwise_affine,
      va.eps};
}

} // namespace FlexFlow
