#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/transform.h"

namespace FlexFlow {

template
  LeafOnlyBinarySeriesSplit<std::string> transform(LeafOnlyBinarySeriesSplit<int> const &, 
                                                   LeafOnlyBinarySPDecompositionTreeVisitor<int, std::string> const &);
template
  LeafOnlyBinaryParallelSplit<std::string> transform(LeafOnlyBinaryParallelSplit<int> const &, 
                                                     LeafOnlyBinarySPDecompositionTreeVisitor<int, std::string> const &);

template
  LeafOnlyBinarySPDecompositionTree<std::string> transform(LeafOnlyBinarySPDecompositionTree<int> const &, 
                                                           LeafOnlyBinarySPDecompositionTreeVisitor<int, std::string> const &);

} // namespace FlexFlow
