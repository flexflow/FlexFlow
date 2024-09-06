#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"

namespace FlexFlow {

BinarySPDecompositionTree make_series_split(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
    GenericBinarySPDecompositionTree<Node>{
      GenericBinarySeriesSplit<Node>{
        lhs.raw_tree,
        rhs.raw_tree
      },
    },
  };
}

BinarySPDecompositionTree make_parallel_split(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
    GenericBinarySPDecompositionTree<Node>{
      GenericBinaryParallelSplit<Node>{
        lhs.raw_tree,
        rhs.raw_tree
      },
    },
  };
}

BinarySPDecompositionTree make_leaf_node(Node const &n) {
  return BinarySPDecompositionTree{
    GenericBinarySPDecompositionTree<Node>{
      n,
    },
  };
}


} // namespace FlexFlow
