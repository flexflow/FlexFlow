#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *dims, size_t num_dims)
    : LegionTensorDims(dims, dims + num_dims) {}

std::size_t ArrayShape::get_volume() const { return product(*this); }

} // namespace FlexFlow
