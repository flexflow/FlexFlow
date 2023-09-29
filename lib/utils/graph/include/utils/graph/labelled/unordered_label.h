#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL

#include "label.h"

namespace FlexFlow {

template <typename Elem, typename Label>
struct UnorderedLabel : virtual public ILabel {
  Label const &get_label(Elem const &e) const {
    return label_map.at(e);
  }

  void add_label(Elem const &e, Label const &l) {
    label_map.insert({e, l});
  }

private:
  std::unordered_map<Elem, Label> label_map;
};

}

#endif
