#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_FLATTENED_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_FLATTENED_DECOMPOSITION_TREE_H

#include "utils/graph/node/node.dtg.h"
#include <unordered_set>
#include <variant>

namespace FlexFlow {

struct SeriesSplit;
struct ParallelSplit;

struct SeriesSplit {
public:
  SeriesSplit() = delete;
  explicit SeriesSplit(std::vector<std::variant<ParallelSplit, Node>> const &);
  explicit SeriesSplit(
      std::initializer_list<std::variant<ParallelSplit, Node>> const &);

  bool operator==(SeriesSplit const &) const;
  bool operator!=(SeriesSplit const &) const;

public:
  std::vector<std::variant<ParallelSplit, Node>> children;

private:
  using Tie = std::tuple<decltype(children) const &>;
  Tie tie() const;
};

std::string format_as(SeriesSplit const &);
std::ostream &operator<<(std::ostream &, SeriesSplit const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::SeriesSplit> {
  size_t operator()(::FlexFlow::SeriesSplit const &) const;
};

} // namespace std

namespace FlexFlow {

struct ParallelSplit {
public:
  ParallelSplit() = delete;
  explicit ParallelSplit(
      std::unordered_set<std::variant<SeriesSplit, Node>> const &);
  explicit ParallelSplit(
      std::initializer_list<std::variant<SeriesSplit, Node>> const &);

  bool operator==(ParallelSplit const &) const;
  bool operator!=(ParallelSplit const &) const;

public:
  std::unordered_set<std::variant<SeriesSplit, Node>> children;

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
