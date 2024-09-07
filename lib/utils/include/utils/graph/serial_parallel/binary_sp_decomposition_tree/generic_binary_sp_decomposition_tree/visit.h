#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_VISIT_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename Result, typename F, typename T>
Result visit(GenericBinarySPDecompositionTree<T> const &tt, F f) {
  if (std::holds_alternative<GenericBinarySeriesSplit<T>>(tt.root)) {
    return f(std::get<GenericBinarySeriesSplit<T>>(tt.root));
  } else if (std::holds_alternative<GenericBinaryParallelSplit<T>>(tt.root)) {
    return f(std::get<GenericBinaryParallelSplit<T>>(tt.root));
  } else if (std::holds_alternative<T>(tt.root)) {
    return f(std::get<T>(tt.root));
  } else {
    throw mk_runtime_error("Unexpected case in visit(GenericBinarySPDecompositionTree)");
  }

  // return std::visit(tt.root, overload {
  //   [&](GenericBinarySeriesSplit<T> const &s) -> Result { 
  //     return f(s);
  //   },
  //   [&](GenericBinaryParallelSplit<T> const &p) -> Result {
  //     return f(p);
  //   },
  //   [&](T const &t) -> Result {
  //     return f(t);
  //   },
  // });
}


} // namespace FlexFlow

#endif
