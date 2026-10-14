#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/set_of.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/minimal_dim_domain.h"

namespace FlexFlow {

positive_int get_total_degree_of_ptensor_dim_degrees(
    ParallelTensorDimDegrees const &degrees) {
  return degrees.sum_degree.value * degrees.discard_copy_degree.value *
         product(degrees.shard_degrees);
}

num_ptensor_shard_dims_t get_ptensor_dim_degrees_num_shard_dims(
    ParallelTensorDimDegrees const &degrees) {
  return num_ptensor_shard_dims_t{
      num_elements(degrees.shard_degrees),
  };
}

num_tensor_dims_t get_ptensor_dim_degrees_num_tensor_dims(
    ParallelTensorDimDegrees const &degrees) {
  return num_tensor_dims_from_num_ptensor_shard_dims(
      get_ptensor_dim_degrees_num_shard_dims(degrees));
}

std::set<parallel_tensor_dim_idx_t>
    get_parallel_tensor_dim_indices(ParallelTensorDimDegrees const &degrees) {

  std::set<parallel_tensor_dim_idx_t> result =
      set_of(dim_idxs_for_num_shard_dims(
          get_ptensor_dim_degrees_num_shard_dims(degrees)));
  result.insert(sum_dim_idx());
  result.insert(discard_copy_dim_idx());
  return result;
}

std::set<parallel_tensor_dim_idx_t> get_nontrivial_parallel_tensor_dim_indices(
    ParallelTensorDimDegrees const &degrees) {
  std::set<parallel_tensor_dim_idx_t> nontrivial_replica_dims;

  if (degrees.sum_degree.value > 1) {
    nontrivial_replica_dims.insert(parallel_tensor_dim_idx_t{ReplicaType::SUM});
  }

  if (degrees.discard_copy_degree.value > 1) {
    nontrivial_replica_dims.insert(
        parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY});
  }

  std::set<parallel_tensor_dim_idx_t> nontrivial_shard_dims = filtrans(
      ff_ordered_get_idxs(degrees.shard_degrees),
      [&](ff_dim_t const &dim) -> std::optional<parallel_tensor_dim_idx_t> {
        if (degrees.shard_degrees.at(dim) > 1) {
          return parallel_tensor_dim_idx_t{dim};
        } else {
          return std::nullopt;
        }
      });

  return set_union(nontrivial_replica_dims, nontrivial_shard_dims);
}

positive_int get_degree_for_parallel_tensor_dim_idx(
    ParallelTensorDimDegrees const &dim_degrees,
    parallel_tensor_dim_idx_t const &idx) {
  if (idx == sum_dim_idx()) {
    return dim_degrees.sum_degree.value;
  } else if (idx == discard_copy_dim_idx()) {
    return dim_degrees.discard_copy_degree.value;
  } else {
    return dim_degrees.shard_degrees.at(idx.require_shard_dim());
  }
}

std::map<parallel_tensor_dim_idx_t, positive_int>
    get_parallel_tensor_degree_map(ParallelTensorDimDegrees const &degrees) {

  std::map<parallel_tensor_dim_idx_t, positive_int> replica_dim_degrees = {
      {parallel_tensor_dim_idx_t{ReplicaType::SUM}, degrees.sum_degree.value},
      {parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY},
       degrees.discard_copy_degree.value},
  };

  std::map<ff_dim_t, positive_int> shard_dim_degrees = generate_map(
      ff_ordered_get_idxs(degrees.shard_degrees),
      [&](ff_dim_t const &dim) { return degrees.shard_degrees.at(dim); });

  return binary_merge_disjoint_maps(
      /*lhs=*/replica_dim_degrees,
      /*rhs=*/map_keys(shard_dim_degrees, [](ff_dim_t const &dim) {
        return parallel_tensor_dim_idx_t{dim};
      }));
}

std::set<ParallelTensorSpaceCoordinate> get_parallel_tensor_space_coordinates(
    ParallelTensorDimDegrees const &degrees) {

  std::map<parallel_tensor_dim_idx_t, positive_int> degree_map =
      get_parallel_tensor_degree_map(degrees);

  std::map<parallel_tensor_dim_idx_t, std::set<nonnegative_int>>
      possible_per_dim_coords = map_values(degree_map, [](positive_int degree) {
        return set_of(nonnegative_range(degree));
      });

  return transform(
      get_all_assignments(possible_per_dim_coords),
      [](std::map<parallel_tensor_dim_idx_t, nonnegative_int> const &m) {
        return parallel_tensor_space_coord_from_map(m);
      });
}

ParallelTensorDimDegrees
    trivial_degrees_for_tensor_dims(TensorDims const &dims) {
  std::vector<positive_int> shard_degrees = repeat_element(
      /*num_times=*/get_num_dims(dims).nonnegative_int_from_num_tensor_dims(),
      /*element=*/1_p);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{1_p},
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/ff_ordered_of(shard_degrees),
  };
}

DimDomain<parallel_tensor_dim_idx_t>
    dim_domain_from_parallel_tensor_dim_degrees(
        ParallelTensorDimDegrees const &dim_degrees) {

  return DimDomain<parallel_tensor_dim_idx_t>{
      generate_map(get_parallel_tensor_dim_indices(dim_degrees),
                   [&](parallel_tensor_dim_idx_t idx) {
                     return get_degree_for_parallel_tensor_dim_idx(dim_degrees,
                                                                   idx);
                   }),
  };
}

ParallelTensorDimDegrees parallel_tensor_dim_degrees_from_dim_domain(
    DimDomain<parallel_tensor_dim_idx_t> const &dim_domain) {

  std::map<ff_dim_t, positive_int> shard_dims =
      filtermap_keys(dim_domain.dims, [](parallel_tensor_dim_idx_t dim_idx) {
        return dim_idx.try_require_shard_dim();
      });

  return ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          dim_domain.dims.at(sum_dim_idx()),
      },
      /*discard_copy_degree=*/
      DiscardCopyDegree{
          dim_domain.dims.at(discard_copy_dim_idx()),
      },
      /*shard_degres=*/ff_ordered_from_map(shard_dims),
  };
}

MinimalDimDomain<parallel_tensor_dim_idx_t>
    minimal_dim_domain_from_parallel_tensor_dim_degrees(
        ParallelTensorDimDegrees const &dim_degrees) {

  return minimal_dim_domain_from_dim_domain(
      dim_domain_from_parallel_tensor_dim_degrees(dim_degrees));
}

} // namespace FlexFlow
