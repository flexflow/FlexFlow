#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"

namespace FlexFlow {

SerialParallelDecomposition
    nary_sp_tree_from_binary(BinarySPDecompositionTree const &binary) {
  return to_final_ast(from_binary_sp_tree(binary));
}

} // namespace FlexFlow
