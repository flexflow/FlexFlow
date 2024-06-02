#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

ArrayShape::ArrayShape(std::vector<std::size_t> const &dims) : dims(dims) {}

std::size_t ArrayShape::get_volume() const { return this->num_elements(); }

std::size_t get_volume(FlexFlow::ArrayShape const &) { NOT_IMPLEMENTED(); }

std::size_t ArrayShape::num_dims() const { return this->dims.size(); }

std::size_t ArrayShape::get_dim() const { return this->num_dims(); }

std::size_t ArrayShape::num_elements() const {
  if (dims.size() == 0)
    return 0;
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<std::size_t>());
}

std::size_t ArrayShape::operator[](legion_dim_t idx) const {
  // necessary to throw out of bounds error?
  return dims[idx];
}

ArrayShape ArrayShape::sub_shape(std::optional<legion_dim_t> start,
                                 std::optional<legion_dim_t> end) {
  NOT_IMPLEMENTED();
}

std::optional<std::size_t> ArrayShape::at_maybe(std::size_t index) const {
  if (index < dims.size()) {
    return dims[legion_dim_t(index)];
  } else {
    return std::nullopt;
  }
}

ArrayShape ArrayShape::reversed_dim_order() const {
  std::vector<std::size_t> reversed_dims(dims.begin(), dims.end());
  std::reverse(reversed_dims.begin(), reversed_dims.end());
  return ArrayShape(reversed_dims);
}

} // namespace FlexFlow
