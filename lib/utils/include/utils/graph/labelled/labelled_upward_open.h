#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H

#include "labelled_upward_open_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledUpwardOpenMultiDiGraphView {
private:
  using Interface = ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>;

public:

private:
  std::shared_ptr<Interface const> ptr;
};

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledDownwardOpenMultiDiGraphView {
private:

public:

private:
};

}

#endif
