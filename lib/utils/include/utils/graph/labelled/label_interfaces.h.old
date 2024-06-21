#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL

#include "utils/graph/open_edge.h"

namespace FlexFlow {

template <typename Elem, typename Label>
struct ILabelling {
  virtual Label const &get_label(Elem const &) const = 0;
  virtual Label &get_label(Elem const &) = 0;
  virtual void add_label(Elem const &, Label const &) = 0;
  virtual ILabelling *clone() const = 0;
};

}; // namespace FlexFlow

#endif
