#include "utils/graph/multidigraph/multidiedge_source.h"

namespace FlexFlow {

size_t MultiDiEdgeSource::next_available_multidiedge_id = 0;

MultiDiEdgeSource::MultiDiEdgeSource() {}

MultiDiEdge MultiDiEdgeSource::new_multidiedge() {
  MultiDiEdge result =
      MultiDiEdge{MultiDiEdgeSource::next_available_multidiedge_id};
  MultiDiEdgeSource::next_available_multidiedge_id++;
  return result;
}

} // namespace FlexFlow
