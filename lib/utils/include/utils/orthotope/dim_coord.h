#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H

#include "utils/containers/all_of.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/map_values.h"
#include "utils/containers/product.h"
#include "utils/containers/require_same.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/scanr.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_domain.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

template <typename T>
std::set<T> get_coord_dims(DimCoord<T> const &coord) {
  return keys(coord.raw);
}

template <typename T>
DimCoord<T> restrict_coord_to_dims(DimCoord<T> const &coord,
                                   std::set<T> const &dims) {
  return DimCoord<T>{
      restrict_keys(coord.raw, dims),
  };
}

template <typename T>
OrthotopeCoord
    orthotope_coord_from_dim_coord(DimCoord<T> const &coord,
                                   DimOrdering<T> const &dim_ordering) {
  return OrthotopeCoord{
      transform(sorted_by(get_coord_dims(coord), dim_ordering.lt),
                [&](T const &t) { return coord.raw.at(t); }),
  };
}

template <typename T>
DimCoord<T> dim_coord_from_orthotope_coord(OrthotopeCoord const &coord,
                                           std::set<T> const &dims,
                                           DimOrdering<T> const &dim_ordering) {
  return DimCoord<T>{
      map_from_keys_and_values(sorted_by(dims, dim_ordering.lt), coord.raw),
  };
}

template <typename T>
DimCoord<T> lift_dim_coord(DimCoord<T> const &coord,
                           std::set<T> const &lifted_dims) {
  ASSERT(is_subseteq_of(get_coord_dims(coord), lifted_dims));

  return DimCoord<T>{
      generate_map(lifted_dims,
                   [&](T const &dim) {
                     if (contains_key(coord.raw, dim)) {
                       return coord.raw.at(dim);
                     } else {
                       return 0_n;
                     }
                   }),
  };
}

template <typename T>
std::set<DimCoord<T>> get_coords_in_dim_domain(DimDomain<T> const &dim_domain) {
  std::map<T, std::set<nonnegative_int>> component_possible_values =
      map_values(dim_domain.dims,
                 [](positive_int component_size) -> std::set<nonnegative_int> {
                   return set_of(nonnegative_range(component_size));
                 });

  std::set<DimCoord<T>> result =
      set_of(transform(get_all_assignments(component_possible_values),
                       [](std::map<T, nonnegative_int> const &assignment) {
                         return DimCoord<T>{
                             assignment,
                         };
                       }));

  ASSERT(all_of(result, [&](DimCoord<T> const &c) -> bool {
    return dim_domain_contains_coord(dim_domain, c);
  }));

  return result;
}

template <typename T>
std::set<DimCoord<T>> get_coords_in_minimal_dim_domain(
    MinimalDimDomain<T> const &minimal_dim_domain) {
  return get_coords_in_dim_domain(lift_minimal_dim_domain(minimal_dim_domain));
}

template <typename T>
DimCoord<T> get_maximum_coord_in_domain(DimDomain<T> const &domain) {
  return DimCoord<T>{
      map_values(domain.dims,
                 [](positive_int dim) -> nonnegative_int {
                   return nonnegative_int{
                       dim.int_from_positive_int() - 1,
                   };
                 }),
  };
}

template <typename T>
DimDomain<T> get_domain_for_maximum_coord(DimCoord<T> const &max_coord) {
  return DimDomain<T>{
      map_values(max_coord.raw,
                 [](nonnegative_int dim) -> positive_int { return dim + 1_p; }),
  };
}

template <typename T>
bool dim_domain_contains_coord(DimDomain<T> const &domain,
                               DimCoord<T> const &coord) {
  ASSERT(get_domain_dims(domain) == get_coord_dims(coord));

  std::set<T> dims =
      require_same(get_domain_dims(domain), get_coord_dims(coord));
  return all_of(dims, [&](T const &dim) {
    return coord.raw.at(dim) < domain.dims.at(dim);
  });
}

template <typename T>
bool minimal_dim_domain_contains_coord(MinimalDimDomain<T> const &domain,
                                       DimCoord<T> const &coord) {
  return dim_domain_contains_coord(lift_minimal_dim_domain(domain), coord);
}

template <typename T>
nonnegative_int flatten_dim_coord(DimCoord<T> const &coord,
                                  DimDomain<T> const &domain,
                                  DimOrdering<T> const &dim_ordering) {
  ASSERT(
      get_coord_dims(coord) == get_domain_dims(domain),
      "flatten_dim_coord expected coord dimensions to match domain dimensions",
      coord,
      domain);

  OrthotopeCoord orthotope_coord =
      orthotope_coord_from_dim_coord(coord, dim_ordering);
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain, dim_ordering);

  return flatten_orthotope_coord(orthotope_coord, orthotope_domain);
}

template <typename T>
DimCoord<T> unflatten_dim_coord(nonnegative_int flattened,
                                DimDomain<T> const &domain,
                                DimOrdering<T> const &dim_ordering) {
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain, dim_ordering);
  OrthotopeCoord orthotope_coord =
      unflatten_orthotope_coord(flattened, orthotope_domain);

  return dim_coord_from_orthotope_coord(
      orthotope_coord, get_domain_dims(domain), dim_ordering);
}

} // namespace FlexFlow

#endif
