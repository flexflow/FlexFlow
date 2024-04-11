#include "op-attrs/tensor_dims.h"
#include "utils/containers.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

size_t dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  FFOrdered<ParallelDim> lifted = {
    transform(as_vector(dims.ff_ordered), [](size_t const &size) { return ParallelDim{size, 1, false}; })
  };
  return {lifted};
}

}
