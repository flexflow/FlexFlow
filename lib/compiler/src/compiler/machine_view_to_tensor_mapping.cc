#include "compiler/machine_view_to_tensor_mapping.h"
#include "compiler/allowed_machine_views.h"
#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "pcg/machine_view_dim_idx_t.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/containers/all_of.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/sorted.h"
#include "utils/containers/zip.h"
#include "utils/exception.h"
namespace FlexFlow {

std::unordered_set<MachineViewToTensorMapping>
    get_all_machine_view_to_tensor_mappings(MachineView const &mv,
                                            ParallelTensorShape const &shape) {
  if (!is_valid_machine_view(mv, shape)) {
    throw mk_runtime_error(
        "Invalid MachineView {} for given ParallelTensorShape {}", mv, shape);
  }
  std::vector<machine_view_dim_idx_t> machine_view_dim_ordering =
      get_machine_view_indices(mv);
  std::unordered_set<parallel_tensor_dim_idx_t> shape_indices =
      get_parallel_tensor_dim_indices(shape);
  shape_indices =
      filter(shape_indices, [&](parallel_tensor_dim_idx_t const &idx) {
        return get_degree(get_parallel_dim_at_idx(shape, idx)) != 1;
      });

  std::unordered_set<MachineViewToTensorMapping> result;
  for (std::vector<parallel_tensor_dim_idx_t> const &tensor_dim_orderings :
       get_all_permutations(shape_indices)) {
    MachineViewToTensorMapping mapping =
        MachineViewToTensorMapping(bidict_from_pairs(
            zip(machine_view_dim_ordering, tensor_dim_orderings)));
    if (is_valid_mapping(mapping, mv, shape)) {
      result.insert(mapping);
    }
  }
  return result;
}

bool is_valid_mapping(MachineViewToTensorMapping const &mapping,
                      MachineView const &mv,
                      ParallelTensorShape const &shape) {
  return all_of(mapping.raw_bidict, [&](auto const pair) {
    int mv_degree = get_side_at_idx(mv, pair.first).num_points.unwrapped;
    int tensor_degree = get_degree(get_parallel_dim_at_idx(shape, pair.second));
    return (tensor_degree == mv_degree);
  });
}
} // namespace FlexFlow
