#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN

#include "node_labelled_open.h"
#include "output_labelled_open_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct OutputLabelledOpenMultiDiGraphView
    : virtual NodeLabelledOpenMultiDiGraphView<NodeLabel>,
      virtual OutputLabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>;

public:
  OutputLabelledOpenMultiDiGraphView(
      OutputLabelledOpenMultiDiGraphView const &) = default;
  OutputLabelledOpenMultiDiGraphView &
      operator=(OutputLabelledOpenMultiDiGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return this->get_ptr().at(n);
  }

  EdgeLabel const &at(InputMultiDiEdge const &i) const {
    return this->get_ptr().at(i);
  }

  EdgeLabel const &at(MultiDiOutput const &o) const {
    return this->get_ptr().at(o);
  }

  template <typename... Ts>
  EdgeLabel const &at(variant<Ts...> const &e) const {
    return visit([&](auto const &e) -> auto const & { return this->at(e); }, e);
  }

  template <typename... Ts>
  EdgeLabel &at(variant<Ts...> const &e) {
    return visit([&](auto const &e) -> auto & { return this->at(e); }, e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 OutputLabelledOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return OutputLabelledOpenMultiDiGraphView(
        make_cow_ptr<BaseImpl>(std::forward<Args>(args)...));
  }

protected:
  using NodeLabelledOpenMultiDiGraphView<
      NodeLabel>::NodeLabelledOpenMultiDiGraphView;

private:
  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};

template <typename NodeLabel, typename EdgeLabel>
struct OutputLabelledOpenMultiDiGraph
    : virtual OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {
private:
  using Interface = IOutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel>;

public:
  OutputLabelledOpenMultiDiGraph() = delete;
  OutputLabelledOpenMultiDiGraph(OutputLabelledOpenMultiDiGraph const &) =
      default;
  OutputLabelledOpenMultiDiGraph &
      operator=(OutputLabelledOpenMultiDiGraph const &) = default;

  Node add_node(NodeLabel const &l) {
    return this->get_ptr().add_node(l);
  }

  NodePort add_node_port() {
    return this->get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return this->get_ptr().at(n);
  }

  void add_label(MultiDiOutput const &o, EdgeLabel const &l) {
    this->get_ptr().add_label(o, l);
  };

  void add_label(InputMultiDiEdge const &e, EdgeLabel const &l) {
    this->get_ptr().add_label(e, l);
  }

  void add_edge(OpenMultiDiEdge const &e) {
    return this->get_ptr().add_edge(e);
  }

  EdgeLabel &at(MultiDiOutput const &o) {
    return this->get_ptr().at(o);
  }

  EdgeLabel &at(InputMultiDiEdge const &e) {
    return this->get_ptr().at(e);
  }

  template <typename... Ts>
  EdgeLabel const &at(variant<Ts...> const &e) const {
    return visit([&](auto const &e) -> auto const & { return this->at(e); }, e);
  }

  template <typename... Ts>
  EdgeLabel &at(variant<Ts...> const &e) {
    return visit([&](auto const &e) -> auto & { return this->at(e); }, e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_ptr().query_nodes(q);
  }
  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return this->get_ptr().query_edges(q);
  }

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 OutputLabelledOpenMultiDiGraph>::type
      create() {
    return OutputLabelledOpenMultiDiGraph(make_cow_ptr<BaseImpl>());
  }

  using OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>::at;

private:
  OutputLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr) : GraphView(ptr) {}

  Interface &get_ptr() {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }
};

template <typename NodeLabel, typename EdgeLabel>
void add_label(OutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel> &g,
               OpenMultiDiEdge const &e,
               EdgeLabel const &l) {
  visit([&](auto const &e) { g.add_label(e, l); }, e);
}

} // namespace FlexFlow

#endif
