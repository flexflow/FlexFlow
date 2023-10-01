#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN

#include "node_labelled.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel>
struct OutputLabelledOpenMultiDiGraphView : virtual NodeLabelledOpenMultiDiGraphView<NodeLabel> {
private:
  using Interface = IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>;

public:
  OutputLabelledOpenMultiDiGraphView() = delete;
  OutputLabelledOpenMultiDiGraphView(OutputLabelledOpenMultiDiGraphView const &) = default;
  OutputLabelledOpenMultiDiGraphView &operator=(OutputLabelledOpenMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return get_ptr()->at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return get_ptr()->at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return get_ptr()->query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &q) const {
    return get_ptr()->query_edges(q);
  }

private:
  OutputLabelledOpenMultiDiGraphView(std::shared_ptr<Interface const> ptr) : NodeLabelledOpenMultiDiGraphView<NodeLabel>(ptr) {}
  std::shared_ptr<Interface const> get_ptr() const {
    return static_assert<std::shared_ptr<Interface const>>(ptr);
  }
};

template <typename NodeLabel,
          typename EdgeLabel>
struct OutputLabelledOpenMultiDiGraph : virtual OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = IMultiDiGraph;
  using INodeLabel = ILabel<Node, NodeLabel>;
  using IEdgeLabel = IOutputOpenLabel<EdgeLabel>;
public:
  OutputLabelledOpenMultiDiGraph() = delete;
  OutputLabelledOpenMultiDiGraph(OutputLabelledOpenMultiDiGraph const &) =
      default;
  OutputLabelledOpenMultiDiGraph &
      operator=(OutputLabelledOpenMultiDiGraph const &) = default;

  Node add_node(NodeLabel const &) {
    NOT_IMPLEMENTED();
  }
  NodeLabel const &at(Node const &) const {
    NOT_IMPLEMENTED();
  }
  NodeLabel &at(Node const &) {
    NOT_IMPLEMENTED();
  }

  void add_edge(MultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  void add_edge(InputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  void add_edge(OutputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }

  InputLabel const &at(InputMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }
  OutputLabel const &at(OutputMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }

  InputLabel &at(InputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  OutputLabel &at(OutputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }

  void add_output(MultiDiOutput const &, EdgeLabel const &) {
    NOT_IMPLEMENTED();
  }
  OutputLabel const &at(MultiDiOutput const &) const {
    NOT_IMPLEMENTED();
  }
  OutputLabel &at(MultiDiOutput const &) {
    NOT_IMPLEMENTED();
  }

  OutputLabel const &at(OpenMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }

  template <typename BaseImpl, typename N, typename E>
  static typename std::enable_if<std::conjunction<std::is_base_of<Interface, BaseImpl>,
                                std::is_base_of<INodeLabel, N>,
                                std::is_base_of<IEdgeLabel, E>>::value
                                OutputLabelledOpenMultiDiGraph>::type
      create() {
    return OutputLabelledOpenMultiDiGraph(
        make_cow_ptr<BaseImpl>(),
        make_cow_ptr<N>(),
        make_cow_ptr<E>());
  }

private:
  OutputLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr, cow_ptr_t<INodeLabel> nl, cow_ptr_t<IEdgeLabel> el)
    : OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>(ptr), nl(nl), el(el) {}
  
  cow_ptr_t<INodeLabel> nl;
  cow_ptr_t<IEdgeLabel> el;
};

} // namespace FlexFlow

#endif
