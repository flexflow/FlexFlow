#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_LABEL

namespace FlexFlow {

template <typename Elem, typename Label>
struct ILabel {
  Label const &get_label(Elem const &) const;
  Label &get_label(Elem const &);
  void add_label(Elem const &, Label const &);
  ILabel *clone() const;
};

template <typename Label>
struct IOutputOpenLabel {
  Label const &get_label(MultiDiOutput const &) const;
  Label &get_label(MultiDiOutput const &);
  Label const &get_label(InputMultiDiEdge const &) const;
  Label &get_label(InputMultiDiEdge const &);
  void add_label(InputMultiDiEdge const &, Label const &);
  void add_label(MultiDiOutput const &, Label const &);
  IOutputOpenLabel *clone() const;
};

};

#endif
