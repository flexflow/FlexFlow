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
   * @param num_dims Number of dimensions
   * @param dims Details of each dimension
   * @param data_type The data type
   */
  ParallelTensorShape(int num_dims,
                      std::vector<ParallelDim> const &dims,
                      DataType data_type);

  int num_dims;                     ///< Number of dimensions
  std::vector<ParallelDim> dims; ///< Details of each dimension
  DataType data_type;               ///< Data type

  bool operator==(ParallelTensorShape const &other) const;
  bool operator!=(ParallelTensorShape const &other) const;

  RecordFormatter as_dot() const;

  size_t get_piece_size() const;
  bool is_valid() const;

  int get_num_replica_dims() const;
  int get_num_replicas() const;

  std::unordered_map<int, int> get_mv_dim_to_tensor_dim_mapping() const;
  std::unordered_map<int, int> get_tensor_dim_to_mv_dim_mapping() const;
};

std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ParallelTensorShape> {
  size_t operator()(FlexFlow::ParallelTensorShape const &) const;
};
} // namespace std


#endif // _OP_META_PARALLEL_TENSOR_SHAPE_H

