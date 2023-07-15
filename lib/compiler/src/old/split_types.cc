#include "split_types.h"

namespace FlexFlow {
namespace PCG {

/*static*/
NonsequenceSplit NonsequenceSplit::sequential() {
  NonsequenceSplit s;
  s.type = SplitType::SEQUENTIAL;
  s.flip_graphs = false;

  return s;
}

/*static*/
NonsequenceSplit NonsequenceSplit::vertical(int param, bool flip_graphs) {
  NonsequenceSplit s;
  s.type = SplitType::VERTICAL;
  s.param = param;
  s.flip_graphs = flip_graphs;

  return s;
}

/*static*/
NonsequenceSplit NonsequenceSplit::horizontal(int param, bool flip_graphs) {
  NonsequenceSplit s;
  s.type = SplitType::HORIZONTAL;
  s.param = param;
  s.flip_graphs = flip_graphs;

  return s;
}

} // namespace PCG
} // namespace FlexFlow
