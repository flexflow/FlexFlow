#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL

#include "label_interfaces.h"
#include "utils/graph/open_edge.h"

namespace FlexFlow {

template <typename Elem, typename Label>
struct UnorderedLabelling : virtual public ILabelling<Elem, Label> {
  UnorderedLabelling() = default;

  Label const &get_label(Elem const &e) const {
    return label_map.at(e);
  }

  Label &get_label(Elem const &e) {
    return label_map.at(e);
  }

  void add_label(Elem const &e, Label const &l) {
    label_map.insert({e, l});
  }

  UnorderedLabelling *clone() const {
    return new UnorderedLabelling(label_map);
  }

private:
  UnorderedLabelling(std::unordered_map<Elem, Label> const &label_map)
      : label_map(label_map) {}
  std::unordered_map<Elem, Label> label_map;
};

} // namespace FlexFlow

#endif
