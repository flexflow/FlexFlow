#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES_H

#include "multidigraph.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct InputMultiDiEdge : public use_visitable_cmp<InputMultiDiEdge> {
  InputMultiDiEdge() = delete;
  InputMultiDiEdge(std::pair<std::size_t, std::size_t> const &,
                   Node const &,
                   NodePort const &);

  std::pair<std::size_t, std::size_t>
      uid; // necessary to differentiate multiple input edges from different
           // sources resulting from a graph cut
  Node dst;
  NodePort dstIdx;
};

struct OutputMultiDiEdge : use_visitable_cmp<OutputMultiDiEdge> {
  OutputMultiDiEdge() = delete;
  OutputMultiDiEdge(std::pair<std::size_t, std::size_t> const &,
                    Node const &,
                    NodePort const &);

  std::pair<std::size_t, std::size_t>
      uid; // necessary to differentiate multiple output edges from different
           // sources resulting from a graph cut
  Node src;
  NodePort srcIdx;
};

using OpenMultiDiEdge =
    variant<InputMultiDiEdge, OutputMultiDiEdge, MultiDiEdge>;

using DownwardOpenMultiDiEdge = variant<OutputMultiDiEdge, MultiDiEdge>;

using UpwardOpenMultiDiEdge = variant<InputMultiDiEdge, MultiDiEdge>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OutputMultiDiEdgeQuery {
  optional<std::unordered_set<Node>> srcs = nullopt;
  optional<std::unordered_set<NodePort>> srcIdxs = nullopt;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};

struct InputMultiDiEdgeQuery {
  optional<std::unordered_set<Node>> dsts = nullopt;
  optional<std::unordered_set<NodePort>> dstIdxs = nullopt;

  static InputMultiDiEdgeQuery all();
  static InputMultiDiEdgeQuery none();
};

struct OpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
  OutputMultiDiEdgeQuery output_edge_query;
};

struct DownwardOpenMultiDiEdgeQuery {
  OutputMultiDiEdgeQuery output_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};

struct UpwardOpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::InputMultiDiEdge, uid, dst, dstIdx);
VISITABLE_STRUCT(::FlexFlow::OutputMultiDiEdge, uid, src, srcIdx);
MAKE_VISIT_HASHABLE(::FlexFlow::InputMultiDiEdge);
MAKE_VISIT_HASHABLE(::FlexFlow::OutputMultiDiEdge);

namespace FlexFlow {

static_assert(is_hashable<OutputMultiDiEdge>::value,
              "OpenMultiDiEdge must be hashable");
static_assert(is_hashable<OpenMultiDiEdge>::value,
              "OpenMultiDiEdge must be hashable");

struct IOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IOpenMultiDiGraphView>::value,
              RC_COPY_VIRTUAL_MSG);

struct IDownwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(
    is_rc_copy_virtual_compliant<IDownwardOpenMultiDiGraphView>::value,
    RC_COPY_VIRTUAL_MSG);

struct IUpwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IUpwardOpenMultiDiGraphView>::value,
              RC_COPY_VIRTUAL_MSG);

struct IOpenMultiDiGraph : public IOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
  virtual IOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IOpenMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

struct IUpwardOpenMultiDiGraph : public IUpwardOpenMultiDiGraphView,
                                 public IGraph {
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual IUpwardOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IUpwardOpenMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

struct IDownwardOpenMultiDiGraph : public IDownwardOpenMultiDiGraphView,
                                   public IGraph {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual IDownwardOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IDownwardOpenMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

} // namespace FlexFlow

#endif
