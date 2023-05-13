#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "ffconst.h"
#include <vector>
#include "utils/record_formatter.h"
#include <unordered_map>
#include "utils/visitable.h"
#include "utils/stack_vector.h"
#include "op-attrs/tensor_shape.h"
#include "parallel_dim.h"

namespace FlexFlow {

static_assert(is_equal_comparable<ParallelDim>::value, "ParallelDim must support ==");
static_assert(is_neq_comparable<ParallelDim>::value, "ParallelDim must support !=");
static_assert(is_lt_comparable<ParallelDim>::value, "ParallelDim must support <");
static_assert(!is_default_constructible<ParallelDim>::value, "ParallelDim must not be default constructible");
static_assert(is_copy_constructible<ParallelDim>::value, "ParallelDim must be copy constructible");

struct ParallelTensorDims : public FFOrdered<ParallelDim> {
  explicit ParallelTensorDims(TensorDims const &);

  size_t get_volume() const; 
};

/**
 * @brief Represent the shape of a ParallelTensor.
 */
struct ParallelTensorShape : public use_visitable_cmp<ParallelTensorShape> {
  ParallelTensorShape() = delete;

  template <typename Dims>
  ParallelTensorShape(Dims const &dims,
                      DataType data_type)
    : dims(dims), data_type(data_type)
  { }

  ParallelTensorShape(TensorShape const &);

  int num_dims() const;

  ParallelDim const &at(ff_dim_t const &) const;
  ParallelDim &at(ff_dim_t const &);
  ParallelDim const &operator[](ff_dim_t const &) const;
  ParallelDim &operator[](ff_dim_t const &);
public:
  ParallelTensorDims dims;
  DataType data_type;
};

TensorShape get_piece_shape(ParallelTensorShape const &);
int get_num_replica_dims(ParallelTensorShape const &);
int get_num_replicas(ParallelTensorShape const &);

bool is_valid(ParallelTensorShape const &);

std::unordered_map<int, int> get_mv_dim_to_tensor_dim_mapping(ParallelTensorShape const &);
std::unordered_map<int, int> get_tensor_dim_to_mv_dim_mapping(ParallelTensorShape const &);



TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &);
std::vector<TensorShape> get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &);

}

VISITABLE_STRUCT(::FlexFlow::ParallelTensorShape, data_type, dims);
MAKE_VISIT_HASHABLE(::FlexFlow::ParallelTensorShape);

#endif 

