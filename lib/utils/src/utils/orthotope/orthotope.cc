#include "utils/orthotope/orthotope.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/product.h"
#include "utils/containers/scanr.h"
#include "utils/containers/set_of.h"
#include "utils/containers/slice.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_with_strict.h"
#include "utils/containers/zip_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"

namespace FlexFlow {

nonnegative_int orthotope_get_num_dims(Orthotope const &orthotope) {
  return num_elements(orthotope.dims);
}

positive_int orthotope_get_volume(Orthotope const &orthotope) {
  return product(orthotope.dims);
}

std::set<OrthotopeCoord>
    get_all_coords_in_orthotope(Orthotope const &orthotope) {
  std::multiset<std::vector<nonnegative_int>> raw_coords =
      cartesian_product(transform(orthotope.dims, [](positive_int dim_size) {
        return nonnegative_range(dim_size);
      }));

  return set_of(
      transform(raw_coords, [](std::vector<nonnegative_int> const &raw_coord) {
        return OrthotopeCoord{raw_coord};
      }));
}

bool orthotope_contains_coord(Orthotope const &orthotope,
                              OrthotopeCoord const &coord) {
  ASSERT(orthotope.dims.size() == coord.raw.size(),
         "orthotope_contains_coord expected orthotope and coord to have the "
         "same number of dims",
         orthotope,
         coord);

  return all_are_true(zip_with(
      coord.raw, orthotope.dims, [](nonnegative_int c, positive_int o) {
        return c < o;
      }));
}

Orthotope
    restrict_orthotope_to_dims(Orthotope const &orthotope,
                               std::set<nonnegative_int> const &allowed_dims) {
  return Orthotope{
      filter_idxs(
          orthotope.dims,
          [&](nonnegative_int idx) { return contains(allowed_dims, idx); }),
  };
}

nonnegative_int flatten_orthotope_coord(OrthotopeCoord const &coord,
                                        Orthotope const &orthotope) {
  ASSERT(orthotope.dims.size() == coord.raw.size(),
         "flatten_orthotope_coord expected orthotope and coord to have the "
         "same number of dims",
         orthotope,
         coord);
  ASSERT(orthotope_contains_coord(orthotope, coord));

  std::vector<positive_int> steps =
      scanr(orthotope.dims, 1_p, [](positive_int r, positive_int accum) {
        return r * accum;
      });

  nonnegative_int result =
      sum(zip_with_strict(coord.raw,
                          slice(steps, 1, std::nullopt),
                          [](nonnegative_int coord_val, positive_int step) {
                            return coord_val * step;
                          }));

  ASSERT(result <= orthotope_get_maximum_offset(orthotope));

  return result;
}

OrthotopeCoord orthotope_get_maximum_coord(Orthotope const &orthotope) {
  return OrthotopeCoord{
      transform(orthotope.dims,
                [](positive_int d) {
                  return nonnegative_int{d.int_from_positive_int() - 1};
                }),
  };
}

nonnegative_int orthotope_get_maximum_offset(Orthotope const &orthotope) {
  return nonnegative_int{product(orthotope.dims).int_from_positive_int() - 1};
}

OrthotopeCoord unflatten_orthotope_coord(nonnegative_int flattened,
                                         Orthotope const &orthotope) {
  ASSERT(flattened <= orthotope_get_maximum_offset(orthotope));

  std::vector<positive_int> steps =
      scanr(orthotope.dims, 1_p, [](positive_int r, positive_int accum) {
        return r * accum;
      });

  OrthotopeCoord result = OrthotopeCoord{
      zip3_with_strict(
          orthotope.dims,
          slice(steps, 1, std::nullopt),
          slice(steps, 0, -1),
          [&](positive_int dim, positive_int step, positive_int next_step) {
            return (flattened % next_step) / step;
          }),
  };

  ASSERT(orthotope_contains_coord(orthotope, result));

  return result;
}

} // namespace FlexFlow
