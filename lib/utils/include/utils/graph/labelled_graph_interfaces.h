#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_GRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_GRAPH_INTERFACES_H

#include "multidigraph.h"
#include "open_graph_interfaces.h"
#include "utils/visitable.h"

namespace FlexFlow {

template <typename NodeLabel>
struct INodeLabelledMultiDiGraph {
public:
  INodeLabelledMultiDiGraph() = default;
  INodeLabelledMultiDiGraph(INodeLabelledMultiDiGraph const &) = delete;
  INodeLabelledMultiDiGraph &
      operator=(INodeLabelledMultiDiGraph const &) = delete;
  virtual ~INodeLabelledMultiDiGraph() {}

  virtual Node add_node(NodeLabel const &) = 0;
  virtual NodeLabel &at(Node const &n) = 0;
  virtual NodeLabel const &at(Node const &n) const = 0;
};

static_assert(
    is_rc_copy_virtual_compliant<INodeLabelledMultiDiGraph<int>>::value,
    RC_COPY_VIRTUAL_MSG);

template <typename NodeLabel, typename EdgeLabel>
struct ILabelledMultiDiGraph : public INodeLabelledMultiDiGraph<NodeLabel>,
                               public IMultiDiGraphView {
  ILabelledMultiDiGraph() = delete;
  ILabelledMultiDiGraph(ILabelledMultiDiGraph const &) = delete;

  virtual ~ILabelledMultiDiGraph();

  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual void add_edge(MultiDiEdge const &) = 0;
  virtual EdgeLabel &at(MultiDiEdge const &);
  virtual EdgeLabel const &at(MultiDiEdge const &) const;
};

static_assert(
    is_rc_copy_virtual_compliant<ILabelledMultiDiGraph<int, int>>::value,
    RC_COPY_VIRTUAL_MSG);

struct MultiDiOutput : public use_visitable_cmp<MultiDiOutput> {
public:
  MultiDiOutput() = delete;
  MultiDiOutput(Node const &node, size_t idx) : node(node), idx(idx) {}

public:
  Node node;
  size_t idx;
};

struct MultiDiInput : public use_visitable_cmp<MultiDiInput> {
public:
  MultiDiInput(Node const &node, size_t idx) : node(node), idx(idx) {}

public:
  Node node;
  size_t idx;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MultiDiOutput, node, idx);
MAKE_VISIT_HASHABLE(::FlexFlow::MultiDiOutput);

VISITABLE_STRUCT(::FlexFlow::MultiDiInput, node, idx);
MAKE_VISIT_HASHABLE(::FlexFlow::MultiDiInput);

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraph
    : public INodeLabelledMultiDiGraph<NodeLabel>,
      public IMultiDiGraphView {
public:
  virtual void add_output(MultiDiOutput const &output,
                          OutputLabel const &label) = 0;
  virtual void add_edge(MultiDiOutput const &output,
                        MultiDiInput const &input) = 0;

  virtual OutputLabel &at(MultiDiOutput const &output) = 0;
  virtual OutputLabel const &at(MultiDiOutput const &output) const = 0;
};

static_assert(
    is_rc_copy_virtual_compliant<IOutputLabelledMultiDiGraph<int, int>>::value,
    RC_COPY_VIRTUAL_MSG);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct ILabelledOpenMultiDiGraph
    : public ILabelledMultiDiGraph<NodeLabel, EdgeLabel> {
public:
  virtual void add_edge(InputMultiDiEdge const &e, InputLabel const &label) = 0;
  virtual void add_edge(OutputMultiDiEdge const &e,
                        OutputLabel const &label) = 0;

  virtual InputLabel const &at(InputMultiDiEdge const &e) const = 0;
  virtual InputLabel &at(InputMultiDiEdge const &e) = 0;

  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const = 0;
  virtual OutputLabel &at(DownwardOpenMultiDiEdge const &e) = 0;
};

static_assert(is_rc_copy_virtual_compliant<
                  ILabelledOpenMultiDiGraph<int, int, int, int>>::value,
              RC_COPY_VIRTUAL_MSG);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::MultiDiOutput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiOutput const &x, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(
        fmt::format("MultiDiOutput({}, {})", x.node, x.idx), ctx);
  }
};

template <>
struct formatter<::FlexFlow::MultiDiInput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiInput const &x, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(
        fmt::format("MultiDiInput({}, {})", x.node, x.idx), ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

static_assert(is_hashable<MultiDiOutput>::value,
              "MultiDiOutput must be hashable");
static_assert(is_equal_comparable<MultiDiOutput>::value,
              "MultiDiOutput must support ==");
static_assert(is_neq_comparable<MultiDiOutput>::value,
              "MultiDiOutput must support !=");
static_assert(is_lt_comparable<MultiDiOutput>::value,
              "MultiDiOutput must support <");
static_assert(!is_default_constructible<MultiDiOutput>::value,
              "MultiDiOutput must not be default constructible");
static_assert(is_copy_constructible<MultiDiOutput>::value,
              "MultiDiOutput must be copy constructible");
static_assert(is_fmtable<MultiDiOutput>::value,
              "MultiDiOutput must support fmt");

} // namespace FlexFlow

#endif
