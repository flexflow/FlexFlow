#include "compiler/series_parallel/computation_graph/computation_graph_binary_sp_decomposition.h"
#include "compiler/series_parallel/computation_graph/get_computation_graph_series_parallel_decomposition.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/overload.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
    ComputationGraphBinarySPDecomposition,
    ComputationGraphBinarySeriesSplit,
    ComputationGraphBinaryParallelSplit,
    layer_guid_t>
    generic_impl_for_computation_graph_sp_tree() {

  return GenericBinarySPDecompositionTreeImplementation<
      ComputationGraphBinarySPDecomposition,
      ComputationGraphBinarySeriesSplit,
      ComputationGraphBinaryParallelSplit,
      layer_guid_t>{
      /*series_get_left_child=*/
      [](ComputationGraphBinarySeriesSplit const &split)
          -> ComputationGraphBinarySPDecomposition const & {
        return split.get_left_child();
      },
      /*parallel_get_left_child=*/
      [](ComputationGraphBinaryParallelSplit const &split)
          -> ComputationGraphBinarySPDecomposition const & {
        return split.get_left_child();
      },
      /*series_get_right_child=*/
      [](ComputationGraphBinarySeriesSplit const &split)
          -> ComputationGraphBinarySPDecomposition const & {
        return split.get_right_child();
      },
      /*parallel_get_right_child=*/
      [](ComputationGraphBinaryParallelSplit const &split)
          -> ComputationGraphBinarySPDecomposition const & {
        return split.get_right_child();
      },
      /*get_node_type=*/
      [](ComputationGraphBinarySPDecomposition const &tree)
          -> SPDecompositionTreeNodeType { return get_node_type(tree); },
      /*require_series=*/
      [](ComputationGraphBinarySPDecomposition const &tree)
          -> ComputationGraphBinarySeriesSplit const & {
        return tree.get<ComputationGraphBinarySeriesSplit>();
      },
      /*require_parallel=*/
      [](ComputationGraphBinarySPDecomposition const &tree)
          -> ComputationGraphBinaryParallelSplit const & {
        return tree.get<ComputationGraphBinaryParallelSplit>();
      },
      /*require_leaf=*/
      [](ComputationGraphBinarySPDecomposition const &tree)
          -> layer_guid_t const & { return tree.get<layer_guid_t>(); },
  };
}

SPDecompositionTreeNodeType
    get_node_type(ComputationGraphBinarySPDecomposition const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload{
      [](ComputationGraphBinarySeriesSplit const &) {
        return SPDecompositionTreeNodeType::SERIES;
      },
      [](ComputationGraphBinaryParallelSplit const &parallel) {
        return SPDecompositionTreeNodeType::PARALLEL;
      },
      [](layer_guid_t const &leaf) {
        return SPDecompositionTreeNodeType::NODE;
      },
  });
}

layer_guid_t require_node(ComputationGraphBinarySPDecomposition const &tree) {
  return tree.get<layer_guid_t>();
}

ComputationGraphBinarySPDecomposition
    computation_graph_sp_decomp_from_binary_sp_decomp(
        BinarySPDecompositionTree const &bin) {
  return bin.visit<ComputationGraphBinarySPDecomposition>(overload{
      [](BinarySeriesSplit const &series) {
        return ComputationGraphBinarySPDecomposition{
            ComputationGraphBinarySeriesSplit{
                computation_graph_sp_decomp_from_binary_sp_decomp(
                    series.get_left_child()),
                computation_graph_sp_decomp_from_binary_sp_decomp(
                    series.get_right_child()),
            },
        };
      },
      [](BinaryParallelSplit const &parallel) {
        return ComputationGraphBinarySPDecomposition{
            ComputationGraphBinaryParallelSplit{
                computation_graph_sp_decomp_from_binary_sp_decomp(
                    parallel.get_left_child()),
                computation_graph_sp_decomp_from_binary_sp_decomp(
                    parallel.get_right_child()),
            },
        };
      },
      [](Node const &node) {
        return ComputationGraphBinarySPDecomposition{
            layer_guid_t{node},
        };
      },
  });
}

std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_left_assoc_binary_sp_decomposition(
        ComputationGraph const &cg) {
  SeriesParallelDecomposition sp_decomposition = ({
    std::optional<SeriesParallelDecomposition> result =
        get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree =
      left_associative_binary_sp_tree_from_nary(sp_decomposition);

  return computation_graph_sp_decomp_from_binary_sp_decomp(raw_binary_tree);
}

std::optional<ComputationGraphBinarySPDecomposition>
    get_computation_graph_right_assoc_binary_sp_decomposition(
        ComputationGraph const &cg) {
  SeriesParallelDecomposition sp_decomposition = ({
    std::optional<SeriesParallelDecomposition> result =
        get_computation_graph_series_parallel_decomposition(cg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  BinarySPDecompositionTree raw_binary_tree =
      right_associative_binary_sp_tree_from_nary(sp_decomposition);

  return computation_graph_sp_decomp_from_binary_sp_decomp(raw_binary_tree);
}

bool is_left_associative(ComputationGraphBinarySPDecomposition const &tree) {
  return is_binary_sp_tree_left_associative(
      tree, generic_impl_for_computation_graph_sp_tree());
}

bool is_right_associative(ComputationGraphBinarySPDecomposition const &tree) {
  return is_binary_sp_tree_right_associative(
      tree, generic_impl_for_computation_graph_sp_tree());
}

std::unordered_multiset<layer_guid_t>
    get_layers(ComputationGraphBinarySPDecomposition const &tree) {
  return get_leaves(tree, generic_impl_for_computation_graph_sp_tree());
}

V1BinarySPDecomposition
    to_v1(ComputationGraphBinarySPDecomposition const &tree,
          bidict<int, layer_guid_t> const &layer_numbering) {
  return tree.visit<V1BinarySPDecomposition>(
      overload{[&](ComputationGraphBinarySeriesSplit const &series) {
                 return V1BinarySPDecomposition{
                     V1BinarySeriesSplit{
                         to_v1(series.get_left_child(), layer_numbering),
                         to_v1(series.get_right_child(), layer_numbering),
                     },
                 };
               },
               [&](ComputationGraphBinaryParallelSplit const &parallel) {
                 return V1BinarySPDecomposition{
                     V1BinaryParallelSplit{
                         to_v1(parallel.get_left_child(), layer_numbering),
                         to_v1(parallel.get_right_child(), layer_numbering),
                     },
                 };
               },
               [&](layer_guid_t const &layer) {
                 return V1BinarySPDecomposition{
                     layer_numbering.at_r(layer),
                 };
               }});
}

} // namespace FlexFlow
