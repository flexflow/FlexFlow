#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN

#include "node_labelled_open.h"
#include "utils/graph/adjacency_openmultidigraph.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct IOutputLabelledOpenMultiDiGraphView
    : virtual INodeLabelledOpenMultiDiGraphView<NodeLabel> {
  virtual EdgeLabel const &at(InputMultiDiEdge const &) const = 0;
  virtual EdgeLabel const &at(MultiDiOutput const &) const = 0;

  using INodeLabelledOpenMultiDiGraphView<NodeLabel>::at;
};

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
    return get_ptr().at(n);
  }

  EdgeLabel const &at(InputMultiDiEdge const &i) const {
    return get_ptr().at(i);
  }

  EdgeLabel const &at(MultiDiOutput const &o) const {
    return get_ptr().at(o);
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
    return get_ptr().query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
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
  using Interface = IOpenMultiDiGraph;
  using INodeLabel = ILabelling<Node, NodeLabel>;
  using IInputLabel = ILabelling<InputMultiDiEdge, EdgeLabel>;
  using IOutputLabel = ILabelling<MultiDiOutput, EdgeLabel>;

public:
  OutputLabelledOpenMultiDiGraph() = delete;
  OutputLabelledOpenMultiDiGraph(OutputLabelledOpenMultiDiGraph const &) =
      default;
  OutputLabelledOpenMultiDiGraph &
      operator=(OutputLabelledOpenMultiDiGraph const &) = default;

  Node add_node(NodeLabel const &l) {
    Node n = get_ptr().add_node();
    this->node_labelling.get_mutable()->add_label(n, l);
    return n;
  }

  void add_node_unsafe(Node const &n, NodeLabel const &l) {
    this->get_ptr().add_node_unsafe(n);
    this->node_labelling.get_mutable()->add_label(n, l);
  }

  NodePort add_node_port() {
    return this->get_ptr().add_node_port();
  }

  NodeLabel &at(Node const &n) {
    return this->node_labelling.get_mutable()->get_label(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->node_labelling->get_label(n);
  }

  void add_label(MultiDiOutput const &o, EdgeLabel const &l) {
    this->output_labelling.get_mutable()->add_label(o, l);
  };

  void add_label(InputMultiDiEdge const &e, EdgeLabel const &l) {
    this->input_labelling.get_mutable()->add_label(e, l);
  }

  void add_edge(OpenMultiDiEdge const &e) {
    return this->get_ptr().add_edge(e);
  }

  EdgeLabel &at(MultiDiOutput const &o) {
    return this->output_labelling.get_mutable()->get_label(o);
  }
  EdgeLabel const &at(MultiDiOutput const &o) const {
    return this->output_labelling->get_label(o);
  }

  EdgeLabel &at(InputMultiDiEdge const &e) {
    return this->input_labelling.get_mutable()->get_label(e);
  }

  EdgeLabel const &at(InputMultiDiEdge const &e) const {
    return this->input_labelling->get_label(e);
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
    return get_ptr().query_nodes(q);
  }
  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const {
    return get_ptr().query_edges(q);
  }

  template <typename BaseImpl, typename N, typename I, typename O>
  static typename std::enable_if<
      std::conjunction<std::is_base_of<Interface, BaseImpl>,
                       std::is_base_of<INodeLabel, N>,
                       std::is_base_of<IInputLabel, I>,
                       std::is_base_of<IOutputLabel, O>>::value,
      OutputLabelledOpenMultiDiGraph>::type
      create() {
    return OutputLabelledOpenMultiDiGraph(make_cow_ptr<BaseImpl>(),
                                          make_cow_ptr<N>(),
                                          make_cow_ptr<I>(),
                                          make_cow_ptr<O>());
  }

private:
  OutputLabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr,
                                 cow_ptr_t<INodeLabel> nl,
                                 cow_ptr_t<IInputLabel> il,
                                 cow_ptr_t<IOutputLabel> ol)
      : GraphView(ptr), node_labelling(nl), input_labelling(il),
        output_labelling(ol) {}

  Interface &get_ptr() {
    return *std::reinterpret_pointer_cast<Interface>(
        GraphView::ptr.get_mutable());
  }

  Interface const &get_ptr() const {
    return *std::reinterpret_pointer_cast<Interface const>(
        GraphView::ptr.get());
  }

  cow_ptr_t<INodeLabel> node_labelling;
  cow_ptr_t<IInputLabel> input_labelling;
  cow_ptr_t<IOutputLabel> output_labelling;
};

template <typename NodeLabel, typename EdgeLabel>
void add_label(OutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel> &g,
               OpenMultiDiEdge const &e,
               EdgeLabel const &l) {
  visit([&](auto const &e) { g.add_label(e, l); }, e);
}

} // namespace FlexFlow

#endif
