#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIEDGE
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIEDGE

#include "node.h"
#include "diedge.h"
#include "node_port.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiDiInput : virtual DiInput {
  NodePort dst_idx;
};
FF_VISITABLE_STRUCT(MultiDiInput, dst, dst_idx);

struct MultiDiOutput : virtual DiOutput {
  NodePort src_idx;
};
FF_VISITABLE_STRUCT(MultiDiOutput, src, src_idx);

using edge_uid_t = std::pair<std::size_t, std::size_t>;

struct InputMultiDiEdge : virtual MultiDiInput {
  edge_uid_t uid; // necessary to differentiate multiple input edges from different
           // sources resulting from a graph cut
};
FF_VISITABLE_STRUCT(InputMultiDiEdge, dst, dst_idx, uid);

struct OutputMultiDiEdge : virtual MultiDiOutput {
  edge_uid_t
      uid; // necessary to differentiate multiple output edges from different
           // sources resulting from a graph cut
};
FF_VISITABLE_STRUCT(OutputMultiDiEdge, src, src_idx, uid);

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

struct MultiDiEdge : virtual MultiDiInput, virtual MultiDiOutput {
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

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::MultiDiOutput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiOutput const &x, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(
        fmt::format("MultiDiOutput({}, {})", x.src, x.src_idx), ctx);
  }
};

template <>
struct formatter<::FlexFlow::MultiDiInput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiInput const &x, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(
        fmt::format("MultiDiInput({}, {})", x.dst, x.dst_idx), ctx);
  }
};

} // namespace fmt

#endif
