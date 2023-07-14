#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_OPEN_GRAPH_INTERFACES_H

#include "multidigraph.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct InputMultiDiEdge {
  std::pair<std::size_t, std::size_t>
      uid; // necessary to differentiate multiple input edges from different
           // sources resulting from a graph cut
  Node dst;
  NodePort dstIdx;
};
FF_VISITABLE_STRUCT(InputMultiDiEdge, uid, dst, dstIdx);

struct OutputMultiDiEdge {
  std::pair<std::size_t, std::size_t>
      uid; // necessary to differentiate multiple output edges from different
           // sources resulting from a graph cut
  Node src;
  NodePort srcIdx;
};
FF_VISITABLE_STRUCT(OutputMultiDiEdge, uid, src, srcIdx);

using OpenMultiDiEdge =
    variant<InputMultiDiEdge, OutputMultiDiEdge, MultiDiEdge>;

using DownwardOpenMultiDiEdge = variant<OutputMultiDiEdge, MultiDiEdge>;

using UpwardOpenMultiDiEdge = variant<InputMultiDiEdge, MultiDiEdge>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OutputMultiDiEdgeQuery {
  query_set<Node> srcs;
  query_set<NodePort> srcIdxs;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(OutputMultiDiEdgeQuery, srcs, srcIdxs);

struct InputMultiDiEdgeQuery {
  query_set<Node> dsts;
  query_set<NodePort> dstIdxs;

  static InputMultiDiEdgeQuery all();
  static InputMultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(InputMultiDiEdgeQuery, dsts, dstIdxs);

struct OpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
  OutputMultiDiEdgeQuery output_edge_query;
};
FF_VISITABLE_STRUCT(OpenMultiDiEdgeQuery,
                    input_edge_query,
                    standard_edge_query,
                    output_edge_query);

struct DownwardOpenMultiDiEdgeQuery {
  OutputMultiDiEdgeQuery output_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};
FF_VISITABLE_STRUCT(DownwardOpenMultiDiEdgeQuery,
                    output_edge_query,
                    standard_edge_query);

struct UpwardOpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};
FF_VISITABLE_STRUCT(UpwardOpenMultiDiEdgeQuery,
                    input_edge_query,
                    standard_edge_query);

struct IOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraphView);

struct IDownwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraphView);

struct IUpwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraphView);

struct IOpenMultiDiGraph : public IOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
  virtual IOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraph);

struct IUpwardOpenMultiDiGraph : public IUpwardOpenMultiDiGraphView,
                                 public IGraph {
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual IUpwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraph);

struct IDownwardOpenMultiDiGraph : public IDownwardOpenMultiDiGraphView,
                                   public IGraph {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual IDownwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraph);

} // namespace FlexFlow

#endif
