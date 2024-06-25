// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/downward_open_pattern_edge.struct.toml
/* proj-data
{
  "generated_from": "c67ec363a91ce090dc538dcf76fa1f12"
}
*/

#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_DOWNWARD_OPEN_PATTERN_EDGE_DTG_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_DOWNWARD_OPEN_PATTERN_EDGE_DTG_H

#include "utils/graph.h"
#include <functional>
#include <tuple>

namespace FlexFlow {
struct DownwardOpenPatternEdge {
  DownwardOpenPatternEdge() = delete;
  explicit DownwardOpenPatternEdge(
      ::FlexFlow::DownwardOpenMultiDiEdge const &raw_edge);

  bool operator==(DownwardOpenPatternEdge const &) const;
  bool operator!=(DownwardOpenPatternEdge const &) const;
  bool operator<(DownwardOpenPatternEdge const &) const;
  bool operator>(DownwardOpenPatternEdge const &) const;
  bool operator<=(DownwardOpenPatternEdge const &) const;
  bool operator>=(DownwardOpenPatternEdge const &) const;
  ::FlexFlow::DownwardOpenMultiDiEdge raw_edge;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::DownwardOpenPatternEdge> {
  size_t operator()(::FlexFlow::DownwardOpenPatternEdge const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_DOWNWARD_OPEN_PATTERN_EDGE_DTG_H