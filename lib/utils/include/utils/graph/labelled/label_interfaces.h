#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL

#include "utils/graph/open_edge.h"

namespace FlexFlow {

template <typename Elem, typename Label>
struct ILabelling {
  Label const &get_label(Elem const &) const;
  Label &get_label(Elem const &);
  void add_label(Elem const &, Label const &);
  ILabelling *clone() const;
};

}; // namespace FlexFlow

#endif
