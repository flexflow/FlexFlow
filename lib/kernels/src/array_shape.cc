#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

std::size_t ArrayShape::get_volume() const {
  return this->num_elements();
}

std::size_t ArrayShape::get_dim() const {
  return this->num_dims();
}

std::size_t ArrayShape::num_elements() const {
  return this->product(this->dims);
}

std::size_t ArrayShape::num_dims() const {
  return this->dims.size();
}

std::size_t ArrayShape::operator[](legion_dim_t idx) const {
  return this->dims.at(idx);
}

std::size_t ArrayShape::at(legion_dim_t idx) const {
  return this->dims.at(idx);
}

legion_dim_t ArrayShape::last_idx() const {
  return legion_dim_t(this->dims.size() - 1);
}

legion_dim_t ArrayShape::neg_idx(int idx) const {
  assert(idx < 0 && "Idx should be negative for negative indexing");
  return legion_dim_t(this->dims.size() + idx);
}

optional<std::size_t> ArrayShape::at_maybe(std::size_t idx) const {
  if (idx < this->dims.size()) {
    return this->dims[legion_dim_t(idx)];
  } else {
    return {};
  }
}

ArrayShape ArrayShape::reversed_dim_order() const {
  std::vector<std::size_t> dims_reversed(this->dims.rbegin(), this->dims.rend());
  return ArrayShape(dims_reversed);
}

ArrayShape ArrayShape::sub_shape(optional<legion_dim_t> start,
                                 optional<legion_dim_t> end) {
  size_t s = start.has_value() ? start.value().value() : 0;
  size_t e = end.has_value() ? end.value().value() : this->dims.size();
  std::vector<std::size_t> sub_dims(this->dims.begin() + s, this->dims.begin() + e);
  return ArrayShape(sub_dims);
}

size_t get_volume(ArrayShape const &shape) {
  return shape.get_volume();
}

} // namespace FlexFlow
