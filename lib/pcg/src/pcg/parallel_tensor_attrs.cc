#include "pcg/parallel_tensor_attrs.h"

namespace FlexFlow {

std::vector<TensorAttrs>
    get_piece_attrs(std::vector<ParallelTensorAttrs> const &parallel_attrs) {
  return transform(parallel_attrs, [](ParallelTensorAttrs const &p) {
    return get_piece_attrs(p);
  });
}

TensorAttrs get_piece_attrs(ParallelTensorAttrs const &parallel_attrs) {
  return {get_piece_shape(parallel_attrs.shape),
          parallel_attrs.initializer,
          parallel_attrs.sync_type,
          parallel_attrs.create_gradients};
}

} // namespace FlexFlow