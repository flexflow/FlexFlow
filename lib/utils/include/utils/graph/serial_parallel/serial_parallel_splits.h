#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIAL_PARALLEL_SPLITS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIAL_PARALLEL_SPLITS_H

#include "utils/graph/node/node.dtg.h"
#include <unordered_set>
#include <variant>

namespace FlexFlow {

struct SerialSplit;
struct ParallelSplit;

struct SerialSplit {
public:
  SerialSplit();
  explicit SerialSplit(std::vector<std::variant<ParallelSplit, Node>> const &);
  explicit SerialSplit(
      std::initializer_list<std::variant<ParallelSplit, Node>> const &);
  explicit SerialSplit(std::vector<Node> const &nodes);

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

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::SerialSplit> {
  size_t operator()(::FlexFlow::SerialSplit const &) const;
};

} // namespace std

namespace FlexFlow {

struct ParallelSplit {
public:
  ParallelSplit();
  explicit ParallelSplit(
      std::unordered_set<std::variant<SerialSplit, Node>> const &);
  explicit ParallelSplit(
      std::initializer_list<std::variant<SerialSplit, Node>> const &);
  explicit ParallelSplit(std::unordered_set<Node> const &nodes);

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
