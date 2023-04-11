#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "ffconst.h"
#include <vector>
#include "utils/record_formatter.h"
#include <unordered_map>
#include "utils/visitable.h"
#include "utils/stack_vector.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

struct ParallelDim {
  ParallelDim() = delete;
  ParallelDim(int size, int degree, int parallel_idx, bool is_replica_dim = false);

  bool operator==(ParallelDim const &) const;
  bool operator!=(ParallelDim const &) const;
  bool operator<(ParallelDim const &) const;

  int size;
  int degree;
  int parallel_idx;
  bool is_replica_dim;
};

static_assert(is_equal_comparable<ParallelDim>::value, "ParallelDim must support ==");
static_assert(is_neq_comparable<ParallelDim>::value, "ParallelDim must support !=");
static_assert(is_lt_comparable<ParallelDim>::value, "ParallelDim must support <");
static_assert(!is_default_constructible<ParallelDim>::value, "ParallelDim must not be default constructible");
static_assert(is_copy_constructible<ParallelDim>::value, "ParallelDim must be copy constructible");

/**
 * @brief Represent the shape of a ParallelTensor.
 */
struct ParallelTensorShape {

  /**
   * @brief Default constructor.
   */
  ParallelTensorShape() = default;

  /**
   * @brief Construct a new ParallelTensorShape object.
   *
   * @param dims Details of each dimension
   * @param data_type The data type
   */
  ParallelTensorShape(std::vector<ParallelDim> const &dims,
                      DataType data_type);

  ParallelTensorShape(TensorShape const &);

  bool operator==(ParallelTensorShape const &other) const;
  bool operator!=(ParallelTensorShape const &other) const;

  RecordFormatter as_dot() const;

  size_t get_piece_size() const;
  bool is_valid() const;

  int get_num_replica_dims() const;
  int get_num_replicas() const;

  std::unordered_map<int, int> get_mv_dim_to_tensor_dim_mapping() const;
  std::unordered_map<int, int> get_tensor_dim_to_mv_dim_mapping() const;

  ParallelDim const &at(int index) const;
  ParallelDim &at(int index);

  size_t size() const;
  size_t num_dims() const;

  using iterator = stack_vector<ParallelDim, MAX_TENSOR_DIM>::iterator;
  using const_iterator = stack_vector<ParallelDim, MAX_TENSOR_DIM>::const_iterator;

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;
  iterator end();
  const_iterator end() const;
  const_iterator cend() const;

public:
  DataType data_type;               ///< Data type
  stack_vector<ParallelDim, MAX_TENSOR_DIM> dims;
};

std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);
}

namespace std {
template <>
struct hash<::FlexFlow::ParallelDim> {
  size_t operator()(::FlexFlow::ParallelDim const &) const;
};

template <>
struct hash<::FlexFlow::ParallelTensorShape> {
  size_t operator()(::FlexFlow::ParallelTensorShape const &) const;
};
} 


#endif 

