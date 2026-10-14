#include "op-attrs/tensor_dim_permutation.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/map_from_ff_ordered.h"
#include "op-attrs/tensor_dims.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/maximum.h"
#include "utils/containers/minimum.h"
#include "utils/containers/permute_with_key.h"
#include "utils/containers/require_same.h"
#include "utils/fmt/set.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

static void check_are_contiguous_from_one(std::set<ff_dim_t> const &idxs) {
  if (idxs.empty()) {
    return;
  }

  ASSERT(minimum(idxs) == ff_dim_t{0_n});
  ASSERT(maximum(idxs) == ff_dim_t{nonnegative_int{idxs.size() - 1}});
}

TensorDimPermutation::TensorDimPermutation(
    bidict<ff_dim_t, ff_dim_t> const &raw)
    : raw(raw) {
  check_are_contiguous_from_one(right_entries(raw));
  check_are_contiguous_from_one(left_entries(raw));
}

bool TensorDimPermutation::operator==(TensorDimPermutation const &other) const {
  return this->tie() == other.tie();
}

bool TensorDimPermutation::operator!=(TensorDimPermutation const &other) const {
  return this->tie() == other.tie();
}

bool TensorDimPermutation::operator<(TensorDimPermutation const &other) const {
  return this->tie() < other.tie();
}

bool TensorDimPermutation::operator>(TensorDimPermutation const &other) const {
  return this->tie() > other.tie();
}

bool TensorDimPermutation::operator<=(TensorDimPermutation const &other) const {
  return this->tie() <= other.tie();
}

bool TensorDimPermutation::operator>=(TensorDimPermutation const &other) const {
  return this->tie() >= other.tie();
}

ff_dim_t TensorDimPermutation::at_l(ff_dim_t l) const {
  return this->raw.at_l(l);
}

ff_dim_t TensorDimPermutation::at_r(ff_dim_t r) const {
  return this->raw.at_r(r);
}

num_tensor_dims_t TensorDimPermutation::num_tensor_dims() const {
  return num_tensor_dims_t{
      num_elements(this->raw),
  };
}

bidict<ff_dim_t, ff_dim_t> const &TensorDimPermutation::as_bidict() const {
  return this->raw;
}

std::tuple<bidict<ff_dim_t, ff_dim_t> const &>
    TensorDimPermutation::tie() const {
  return std::tie(this->raw);
}

bidict<ff_dim_t, ff_dim_t> format_as(TensorDimPermutation const &p) {
  return p.as_bidict();
}

std::ostream &operator<<(std::ostream &s, TensorDimPermutation const &p) {
  return (s << fmt::to_string(p));
}

TensorDimPermutation
    compose_tensor_dim_permutations(TensorDimPermutation const &lhs,
                                    TensorDimPermutation const &rhs) {

  ASSERT(lhs.num_tensor_dims() == rhs.num_tensor_dims());

  return TensorDimPermutation{
      exhaustive_relational_join(lhs.as_bidict(), rhs.as_bidict()),
  };
}

TensorDimPermutation
    invert_tensor_dim_permutation(TensorDimPermutation const &p) {

  return TensorDimPermutation{
      p.as_bidict().reversed(),
  };
}

template <typename T>
static FFOrdered<T> permute_ff_ordered(TensorDimPermutation const &permutation,
                                       FFOrdered<T> const &ff_ordered) {
  return ff_ordered_from_map(
      map_keys(map_from_ff_ordered(ff_ordered),
               [&](ff_dim_t k) { return permutation.at_r(k); }));
}

TensorDims permute_tensor_dims(TensorDimPermutation const &permutation,
                               TensorDims const &dims) {
  ASSERT(permutation.num_tensor_dims() == get_num_dims(dims));

  return TensorDims{
      permute_ff_ordered(permutation, dims.ff_ordered),
  };
}

TensorShape permute_tensor_shape(TensorDimPermutation const &permutation,
                                 TensorShape const &shape) {
  return TensorShape{
      /*dims=*/permute_tensor_dims(permutation, shape.dims),
      /*data_type=*/shape.data_type,
  };
}

TensorDimsCoord
    permute_tensor_dims_coord(TensorDimPermutation const &permutation,
                              TensorDimsCoord const &coord) {
  return TensorDimsCoord{
      /*ff_ordered=*/permute_ff_ordered(permutation, coord.ff_ordered),
  };
}

ParallelTensorDimDegrees permute_parallel_tensor_dim_degrees(
    TensorDimPermutation const &permutation,
    ParallelTensorDimDegrees const &parallel_tensor_dim_degrees) {
  return ParallelTensorDimDegrees{
      /*sum_degree=*/parallel_tensor_dim_degrees.sum_degree,
      /*discard_copy_degree=*/parallel_tensor_dim_degrees.discard_copy_degree,
      /*shard_degrees=*/
      permute_ff_ordered(permutation,
                         parallel_tensor_dim_degrees.shard_degrees),
  };
}

ParallelTensorDims permute_parallel_tensor_dims(
    TensorDimPermutation const &permutation,
    ParallelTensorDims const &parallel_tensor_dims) {
  return ParallelTensorDims{
      /*shard_dims=*/permute_ff_ordered(permutation,
                                        parallel_tensor_dims.shard_dims),
      /*replica_dims=*/parallel_tensor_dims.replica_dims,
  };
}

ParallelTensorShape permute_parallel_tensor_shape(
    TensorDimPermutation const &permutation,
    ParallelTensorShape const &parallel_tensor_shape) {
  return ParallelTensorShape{
      /*dims=*/permute_parallel_tensor_dims(permutation,
                                            parallel_tensor_shape.dims),
      /*data_type=*/parallel_tensor_shape.data_type,
  };
}

ParallelTensorSpaceCoordinate permute_parallel_tensor_space_coordinate(
    TensorDimPermutation const &permutation,
    ParallelTensorSpaceCoordinate const &pt_coord) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/pt_coord.sum_component,
      /*discover_copy_component=*/pt_coord.discard_copy_component,
      /*shard_components=*/
      permute_ff_ordered(permutation, pt_coord.shard_components),
  };
}

} // namespace FlexFlow

namespace nlohmann {

::FlexFlow::TensorDimPermutation
    adl_serializer<::FlexFlow::TensorDimPermutation>::from_json(json const &j) {
  ::FlexFlow::bidict<::FlexFlow::ff_dim_t, ::FlexFlow::ff_dim_t> b = j;

  return ::FlexFlow::TensorDimPermutation{b};
}

void adl_serializer<::FlexFlow::TensorDimPermutation>::to_json(
    json &j, ::FlexFlow::TensorDimPermutation const &p) {
  j = p.as_bidict();
}

} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::TensorDimPermutation>
    Arbitrary<::FlexFlow::TensorDimPermutation>::arbitrary() {
  using namespace ::FlexFlow;

  Gen<std::vector<ff_dim_t>> key_permutation_gen = gen::withSize([=](int size) {
    nonnegative_int reduced_size = std::min(nonnegative_int{size}, 5_n);
    std::vector<ff_dim_t> sized_keys = ff_dim_range(reduced_size);
    return gen::map(gen::arbitrary<int>(),
                    [=](int key) -> std::vector<ff_dim_t> {
                      return permute_with_key(key, sized_keys);
                    });
  });

  return gen::construct<TensorDimPermutation>(gen::apply(
      [](std::vector<ff_dim_t> const &ks, std::vector<ff_dim_t> const &vs) {
        return bidict_from_keys_and_values(ks, vs);
      },
      key_permutation_gen,
      key_permutation_gen));
}

} // namespace rc

namespace std {

size_t hash<::FlexFlow::TensorDimPermutation>::operator()(
    ::FlexFlow::TensorDimPermutation const &p) const {
  return get_std_hash(p.tie());
}

} // namespace std
