#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

ArrayShape::ArrayShape(std::vector<std::size_t> const &vec)
    : dims(vec.begin(), vec.end()) {}

std::size_t ArrayShape::get_volume() const {
  return num_elements();
}

std::size_t ArrayShape::get_dim() const {
  return num_dims();
}

std::size_t ArrayShape::num_elements() const {
  return product(this->dims);
}

std::size_t ArrayShape::num_dims() const {
  return this->dims.size();
}

std::size_t ArrayShape::operator[](legion_dim_t idx) const {
  return dims[idx];
}

std::size_t ArrayShape::at(legion_dim_t idx) const {
  return dims.at(idx);
}

legion_dim_t ArrayShape::last_idx() const {
  return dims.size() - 1;
}

legion_dim_t ArrayShape::neg_idx(int idx) const {
  assert(idx < 0 && "Idx should be negative for negative indexing");
  return dims.size() + idx;
}

optional<std::size_t> ArrayShape::at_maybe(std::size_t idx) const {
  if (idx < dims.size()) {
    return dims[idx];
  } else {
    return {};
  }
}

ArrayShape ArrayShape::reversed_dim_order() const {
  return ArrayShape(dims.rbegin(), dims.rend());
}

ArrayShape ArrayShape::sub_shape(optional<legion_dim_t> start,
                                 optional<legion_dim_t> end) {
  size_t s = start.has_value() ? start.value() : 0;
  size_t e = end.has_value() ? end.value() : dims.size();
  return {LegionTensorDims(dims.begin() + s, dims.begin() + e)};
}

std::size_t ArrayShape::get_volume() const {
  return product(this->dims);
}

bool ArrayShape::operator==(ArrayShape const &other) const {
  if (this->dims.size() != other.dims.size()) {
    return false;
  }

  return this->dims == other.dims;
}

} // namespace FlexFlow
