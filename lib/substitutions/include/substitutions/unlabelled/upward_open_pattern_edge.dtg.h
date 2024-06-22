// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/upward_open_pattern_edge.struct.toml
/* proj-data
{
  "generated_from": "a1d4c9d1dd94eb456c5e29d80ad579da"
}
*/

#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UPWARD_OPEN_PATTERN_EDGE_DTG_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UPWARD_OPEN_PATTERN_EDGE_DTG_H

#include "utils/graph.h"
#include <functional>
#include <tuple>

namespace FlexFlow {
struct UpwardOpenPatternEdge {
  UpwardOpenPatternEdge() = delete;
  explicit UpwardOpenPatternEdge(
      ::FlexFlow::UpwardOpenMultiDiEdge const &raw_edge);

  bool operator==(UpwardOpenPatternEdge const &) const;
  bool operator!=(UpwardOpenPatternEdge const &) const;
  bool operator<(UpwardOpenPatternEdge const &) const;
  bool operator>(UpwardOpenPatternEdge const &) const;
  bool operator<=(UpwardOpenPatternEdge const &) const;
  bool operator>=(UpwardOpenPatternEdge const &) const;
  ::FlexFlow::UpwardOpenMultiDiEdge raw_edge;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::UpwardOpenPatternEdge> {
  size_t operator()(::FlexFlow::UpwardOpenPatternEdge const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UPWARD_OPEN_PATTERN_EDGE_DTG_H
