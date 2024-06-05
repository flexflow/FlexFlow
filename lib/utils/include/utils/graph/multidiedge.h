#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIEDGE
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIEDGE

#include "diedge.h"
#include "node.h"
#include "node_port.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiDiInput : DiInput {
  NodePort dst_idx;
};
FF_VISITABLE_STRUCT(MultiDiInput, dst, dst_idx);
FF_VISIT_FMTABLE(MultiDiInput);

struct MultiDiOutput : DiOutput {
  NodePort src_idx;

  bool operator>(MultiDiOutput const &) const;
  bool operator>=(MultiDiOutput const &) const;
  bool operator<=(MultiDiOutput const &) const;
};
FF_VISITABLE_STRUCT(MultiDiOutput, src, src_idx);
FF_VISIT_FMTABLE(MultiDiOutput);

using edge_uid_t = std::pair<std::size_t, std::size_t>;

struct InputMultiDiEdge : MultiDiInput {
  req<edge_uid_t> uid; // necessary to differentiate multiple input edges from
                       // different sources resulting from a graph cut
};
FF_VISITABLE_STRUCT(InputMultiDiEdge, dst, dst_idx, uid);
FF_VISIT_FMTABLE(InputMultiDiEdge);

struct OutputMultiDiEdge : MultiDiOutput {
  req<edge_uid_t> uid; // necessary to differentiate multiple output edges from
                       // different sources resulting from a graph cut
};
FF_VISITABLE_STRUCT(OutputMultiDiEdge, src, src_idx, uid);
FF_VISIT_FMTABLE(OutputMultiDiEdge);

struct OutputMultiDiEdgeQuery {
  query_set<Node> srcs;
  query_set<NodePort> srcIdxs;

  OutputMultiDiEdgeQuery with_src_nodes(query_set<Node> const &) const;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(OutputMultiDiEdgeQuery, srcs, srcIdxs);

struct InputMultiDiEdgeQuery {
  query_set<Node> dsts;
  query_set<NodePort> dstIdxs;

  InputMultiDiEdgeQuery with_dst_nodes(query_set<Node> const &) const;

  static InputMultiDiEdgeQuery all();
  static InputMultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(InputMultiDiEdgeQuery, dsts, dstIdxs);

struct MultiDiEdge : MultiDiInput, MultiDiOutput {
  edge_uid_t get_uid() const {
    return std::make_pair(src_idx.value(), dst_idx.value());
  }
};
FF_VISITABLE_STRUCT(MultiDiEdge, dst, dst_idx, src, src_idx);
FF_VISIT_FMTABLE(MultiDiEdge);

struct MultiDiEdgeQuery {
  query_set<Node> srcs;
  query_set<Node> dsts;
  query_set<NodePort> srcIdxs;
  query_set<NodePort> dstIdxs;

  MultiDiEdgeQuery with_src_nodes(query_set<Node> const &) const;
  MultiDiEdgeQuery with_dst_nodes(query_set<Node> const &) const;
  MultiDiEdgeQuery with_src_idxs(query_set<NodePort> const &) const;
  MultiDiEdgeQuery with_dst_idxs(query_set<NodePort> const &) const;

  static MultiDiEdgeQuery all();
  static MultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(MultiDiEdgeQuery, srcs, dsts, srcIdxs, dstIdxs);

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &,
                                    MultiDiEdgeQuery const &);
MultiDiEdgeQuery query_union(MultiDiEdgeQuery const &,
                             MultiDiEdgeQuery const &);

InputMultiDiEdge to_inputmultidiedge(MultiDiEdge const &e);
OutputMultiDiEdge to_outputmultidiedge(MultiDiEdge const &e);

} // namespace FlexFlow

#endif
