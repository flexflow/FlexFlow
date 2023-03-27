#ifndef _FLEXFLOW_KERNELS_ARRAY_SHAPE_H
#define _FLEXFLOW_KERNELS_ARRAY_SHAPE_H

#include <vector>
#include <cstddef>
#include "utils/stack_vector.h"
#include "utils/optional.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ArrayShape {
public:
  ArrayShape(size_t *dims, size_t num_dims);
  ArrayShape(std::vector<std::size_t> const &);

  /**
   * @brief Alias of ArrayShape::num_elements for compatibility with Legion::Domain
   */
  std::size_t get_volume() const;

  /**
   * @brief Alias of ArrayShape::num_dims for compatibility with Legion::Domain
   */
  std::size_t get_dim() const;

  std::size_t num_elements() const;
  std::size_t num_dims() const;

  std::size_t operator[](std::size_t) const;

  optional<std::size_t> at_maybe(std::size_t) const;

  ArrayShape reversed_dim_order() const;
  ArrayShape sub_shape(optional<std::size_t> start, optional<std::size_t> end);

  using iterator = typename stack_vector<std::size_t, MAX_TENSOR_DIM>::iterator;
  using const_iterator = typename stack_vector<std::size_t, MAX_TENSOR_DIM>::const_iterator;

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;

  iterator end();
  const_iterator end() const;
  const_iterator cend() const;

  bool operator==(ArrayShape const &) const;
  bool operator!=(ArrayShape const &) const;

public:
  stack_vector<std::size_t, MAX_TENSOR_DIM> dims;
};

}

VISITABLE_STRUCT(::FlexFlow::ArrayShape, dims);

namespace std {

template <>
struct hash<::FlexFlow::ArrayShape> {
  size_t operator()(::FlexFlow::ArrayShape const &) const;
};

}

#endif 
