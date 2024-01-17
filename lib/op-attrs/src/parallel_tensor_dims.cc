#include "op-attrs/parallel_tensor_dims.h"

namespace FlexFlow {

static std::vector<ParallelDim> lift_dims(TensorDims const &dims) {
  std::vector<ParallelDim> lifted_dims;
  for (size_t dim_size : dims) {
    lifted_dims.push_back({dim_size, 1, false});
  }
  lifted_dims.push_back({1, 1, true});
  return lifted_dims;
}

ParallelTensorDims::ParallelTensorDims(TensorDims const &dims)
    : data(lift_dims(dims)) {}

ParallelDim ParallelTensorDims::at(ff_dim_t const &dim) const {
  return data.at(dim);
}

size_t ParallelTensorDims::num_dims() const {
  return data.num_dims();
}

size_t ParallelTensorDims::get_volume() const {
  size_t volume = 1;
  for (int i = 0; i < num_dims(); i++) {
    volume *= at(ff_dim_t(i)).size;
  }

  return volume;
}

TensorDims get_piece_dims(ParallelTensorDims const &dims){NOT_IMPLEMENTED()}

TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &dim) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
