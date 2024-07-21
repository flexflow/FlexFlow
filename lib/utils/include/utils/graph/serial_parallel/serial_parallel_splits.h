#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_FLATTENED_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_FLATTENED_DECOMPOSITION_TREE_H

#include "utils/graph/node/node.dtg.h"
#include <variant>
#include <unordered_set>

namespace FlexFlow {

struct SerialSplit;
struct ParallelSplit;

struct SerialSplit {
public:
  SerialSplit() = delete;
  explicit SerialSplit(std::vector<std::variant<ParallelSplit, Node>> const &);

  bool operator==(SerialSplit const &) const;
  bool operator!=(SerialSplit const &) const;
public:
  std::vector<std::variant<ParallelSplit, Node>> children;

private:
  using Tie = std::tuple<decltype(children) const &>;
  Tie tie() const;
};

std::string format_as(SerialSplit const &);
std::ostream &operator<<(std::ostream &, SerialSplit const &);

}

namespace std {

template <>
struct hash<::FlexFlow::SerialSplit> {
  size_t operator()(::FlexFlow::SerialSplit const &) const;
};

}

namespace FlexFlow {

struct ParallelSplit {
public:
  ParallelSplit() = delete;
  explicit ParallelSplit(std::unordered_set<std::variant<SerialSplit, Node>> const &);

  bool operator==(ParallelSplit const &) const;
  bool operator!=(ParallelSplit const &) const;
public:
  std::unordered_set<std::variant<SerialSplit, Node>> children;

private:
  using Tie = std::tuple<decltype(children) const &>;
  Tie tie() const;
};

std::string format_as(ParallelSplit const &);
std::ostream &operator<<(std::ostream &, ParallelSplit const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ParallelSplit> {
  size_t operator()(::FlexFlow::ParallelSplit const &) const;
};

} // namespace std

#endif
