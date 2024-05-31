// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/edge_splits.struct.toml
/* proj-data
{
  "generated_from": "f172b041a99f4de1d396e5d451a5e64d"
}
*/

#include "substitutions/unlabelled/edge_splits.dtg.h"

#include "utils/bidict.h"
#include "utils/graph.h"
#include <utility>

namespace FlexFlow {
UnlabelledPatternEdgeSplits::UnlabelledPatternEdgeSplits(
    ::FlexFlow::bidict<::FlexFlow::MultiDiEdge,
                       std::pair<::FlexFlow::OutputMultiDiEdge,
                                 ::FlexFlow::InputMultiDiEdge>> const
        &unwrapped)
    : unwrapped(unwrapped) {}
bool UnlabelledPatternEdgeSplits::operator==(
    UnlabelledPatternEdgeSplits const &other) const {
  return std::tie(this->unwrapped) == std::tie(other.unwrapped);
}
bool UnlabelledPatternEdgeSplits::operator!=(
    UnlabelledPatternEdgeSplits const &other) const {
  return std::tie(this->unwrapped) != std::tie(other.unwrapped);
}
} // namespace FlexFlow