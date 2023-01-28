#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "ffconst.h"
#include <vector>
#include "utils/record_formatter.h"
#include <unordered_map>

namespace FlexFlow {

struct ParallelDim {
  static constexpr int UNKNOWN_DEGREE = -1;
  static constexpr int UNKNOWN_INDEX = -2;

  bool operator==(ParallelDim const &rhs) const {
    if (size != rhs.size) {
      return false;
    }
    if (degree != rhs.degree) {
      return false;
    }
    if (parallel_idx != rhs.parallel_idx) {
      return false;
    }
    return true;
  }

  bool operator!=(ParallelDim const &rhs) const {
    if (size != rhs.size) {
      return true;
    }
    if (degree != rhs.degree) {
      return true;
    }
    if (parallel_idx != rhs.parallel_idx) {
      return true;
    }
    return false;
  }


  using AsConstTuple = std::tuple<int, int, int, bool>;
  AsConstTuple as_tuple() const;

public:
  int size = 0;
  int degree = UNKNOWN_DEGREE;
  int parallel_idx = UNKNOWN_INDEX;
  bool is_replica_dim = false;
};


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

  using iterator = std::vector<ParallelDim>::iterator;
  using const_iterator = std::vector<ParallelDim>::const_iterator;

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;
  iterator end();
  const_iterator end() const;
  const_iterator cend() const;

public:
  DataType data_type;               ///< Data type
private:
  std::vector<ParallelDim> dims; ///< Details of each dimension
};

std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ParallelDim> {
  size_t operator()(FlexFlow::ParallelDim const &) const;
};

template <>
struct hash<FlexFlow::ParallelTensorShape> {
  size_t operator()(FlexFlow::ParallelTensorShape const &) const;
};
} // namespace std


#endif // _OP_META_PARALLEL_TENSOR_SHAPE_H

