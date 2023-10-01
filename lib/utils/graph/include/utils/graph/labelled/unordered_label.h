#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_UNORDERED_LABEL

#include "label.h"
#include "utils/graph/open_edge.h"

namespace FlexFlow {

template <typename Elem, typename Label>
struct UnorderedLabel : virtual public ILabel<Elem, Label> {
  UnorderedLabel() = default;

  Label const &get_label(Elem const &e) const {
    return label_map.at(e);
  }

  Label &get_label(Elem const &e) {
    return label_map[e];
  }

  void add_label(Elem const &e, Label const &l) {
    label_map.insert({e, l});
  }

  UnorderedLabel *clone() const {
    return new UnorderedLabel(label_map);
  }

private:
  UnorderedLabel(std::unordered_map<Elem, Label> const &label_map) : label_map(label_map) {}
  std::unordered_map<Elem, Label> label_map;
};

template <typename Label>
struct UnorderedOutputOpenLabel<Label> : virtual public IOutputOpenLabel<Label> {
public:
  UnorderedOutputOpenLabel() = default;

  Label const &get_label(OpenMultiDiEdge const &e) const {
    return visit([&](auto const &e) { return get_label_m(e); }, e);
  }

  Label &get_label(OpenMultiDiEdge const &e) {
    return visit([&](auto const &e) -> Label &{ return get_label_m(e); }, e);
  }

  void add_label(Elem const &e, Label const &l) {
    visit([&](auto const &e) { add_label_m(e); }, e);
  }

  UnorderedLabel *clone() const {
    return new UnorderedLabel<OpenMultiDiEdge, Label>(input_map, output_map);
  }

private:
  Label const &get_label_m(InputMultiDiEdge const &e) const {
    return input_map.at(e);
  }

  Label const &get_label_m(MultiDiEdge const &e) const {
    return output_map.at(e);
  }

  Label const &get_label_m(OutputMultiDiEdge const &e) const {
    return output_map.at(e);
  }

  Label &get_label_m(InputMultiDiEdge const &e) {
    return input_map.at(e);
  }

  Label &get_label_m(MultiDiEdge const &e) {
    return output_map.at(e);
  }

  Label &get_label_m(OutputMultiDiEdge const &e) {
    return output_map.at(e);
  }

  void add_label_m(InputMultiDiEdge const &e, Label const &l) {
    input_map.insert({e, l});
  }

  void add_label_m(MultiDiEdge const &e, Label const &l) {
    output_map.insert({MultiDiOutput(e), l});
  }

  void add_label_m(OutputMultiDiEdge cosnt &e, Label const &l) {
    output_map.insert({MultiDiOutput(e), l});
  }

  UnorderedLabel(std::unordered_map<InputMultiDiEdge, Label> const &input_map,
                 std::unordered_map<MultiDiOutput, Label> const &output_map)
    : input_map(input_map), output_map(output_map) {}

  std::unordered_map<InputMultiDiEdge, Label> input_map;
  std::unordered_map<MultiDiOutput, Label> output_map;
};

}

#endif
