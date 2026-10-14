#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIM_PERMUTATION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIM_PERMUTATION_H

#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "op-attrs/tensor_dims_coord.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

struct TensorDimPermutation {
  TensorDimPermutation() = delete;

  TensorDimPermutation(bidict<ff_dim_t, ff_dim_t> const &);

  bool operator==(TensorDimPermutation const &) const;
  bool operator!=(TensorDimPermutation const &) const;

  bool operator<(TensorDimPermutation const &) const;
  bool operator>(TensorDimPermutation const &) const;
  bool operator<=(TensorDimPermutation const &) const;
  bool operator>=(TensorDimPermutation const &) const;

  ff_dim_t at_l(ff_dim_t) const;
  ff_dim_t at_r(ff_dim_t) const;

  num_tensor_dims_t num_tensor_dims() const;

  bidict<ff_dim_t, ff_dim_t> const &as_bidict() const;

private:
  bidict<ff_dim_t, ff_dim_t> raw;

private:
  std::tuple<decltype(raw) const &> tie() const;

  friend struct std::hash<TensorDimPermutation>;
};

bidict<ff_dim_t, ff_dim_t> format_as(TensorDimPermutation const &);
std::ostream &operator<<(std::ostream &, TensorDimPermutation const &);

TensorDimPermutation
    compose_tensor_dim_permutations(TensorDimPermutation const &,
                                    TensorDimPermutation const &);

TensorDimPermutation
    invert_tensor_dim_permutation(TensorDimPermutation const &);

TensorDims permute_tensor_dims(TensorDimPermutation const &,
                               TensorDims const &);

TensorShape permute_tensor_shape(TensorDimPermutation const &,
                                 TensorShape const &);

TensorDimsCoord permute_tensor_dims_coord(TensorDimPermutation const &,
                                          TensorDimsCoord const &);

ParallelTensorDimDegrees
    permute_parallel_tensor_dim_degrees(TensorDimPermutation const &,
                                        ParallelTensorDimDegrees const &);

ParallelTensorDims permute_parallel_tensor_dims(TensorDimPermutation const &,
                                                ParallelTensorDims const &);

ParallelTensorShape permute_parallel_tensor_shape(TensorDimPermutation const &,
                                                  ParallelTensorShape const &);

ParallelTensorSpaceCoordinate permute_parallel_tensor_space_coordinate(
    TensorDimPermutation const &, ParallelTensorSpaceCoordinate const &);

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::TensorDimPermutation> {
  static ::FlexFlow::TensorDimPermutation from_json(json const &);
  static void to_json(json &, ::FlexFlow::TensorDimPermutation const &);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::TensorDimPermutation> {
  static Gen<::FlexFlow::TensorDimPermutation> arbitrary();
};

} // namespace rc

namespace std {

template <>
struct hash<::FlexFlow::TensorDimPermutation> {
  size_t operator()(::FlexFlow::TensorDimPermutation const &) const;
};

} // namespace std

#endif
