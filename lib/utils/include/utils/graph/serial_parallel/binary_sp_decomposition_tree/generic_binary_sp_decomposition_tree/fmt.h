#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FMT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FMT_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename T>
std::string format_as(GenericBinarySeriesSplit<T> const &s) {
  return fmt::format(
      "<GenericBinarySeriesSplit {} {}>", get_left_child(s), get_right_child(s));
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinarySeriesSplit<T> const &x) {
  return (s << fmt::to_string(x));
}


template <typename T>
std::string format_as(GenericBinaryParallelSplit<T> const &s) {
  return fmt::format(
      "<GenericBinaryParallelSplit {} {}>", get_left_child(s), get_right_child(s));
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinaryParallelSplit<T> const &x) {
  return (s << fmt::to_string(x));
}


template <typename T>
std::string format_as(GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<std::string>(tt, overload{
      [](GenericBinarySeriesSplit<T> const &s) {
        return fmt::format("<GenericBinarySPDecompositionTree {}>", s);
      },
      [](GenericBinaryParallelSplit<T> const &s) {
        return fmt::format("<GenericBinarySPDecompositionTree {}>", s);
      },
      [](T const &t) {
        return fmt::format("<BinarySPDecompositionTree {}>", t);
      },
  });
}

template <typename T>
std::ostream &operator<<(std::ostream &s,
                         GenericBinarySPDecompositionTree<T> const &t) {
  return (s << fmt::to_string(t));
}

} // namespace FlexFlow

#endif
