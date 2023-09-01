#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIEDGE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIEDGE_H

#include "node.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

/**
 * @class NodePort
 * @brief An opaque object used to disambiguate multiple edges between the same
 * nodes in a MultiDiGraph
 *
 * Name chosen to match the terminology used by <a href="linkURL">ELK</a>
 *
 */
struct NodePort : public strong_typedef<NodePort, size_t> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(NodePort);
FF_TYPEDEF_PRINTABLE(NodePort, "NodePort");

struct MultiDiEdge {
  Node src, dst;
  NodePort srcIdx, dstIdx;
};
FF_VISITABLE_STRUCT(MultiDiEdge, src, dst, srcIdx, dstIdx);
FF_VISIT_FMTABLE(MultiDiEdge);

struct MultiDiInput {
  Node node;
  NodePort idx;
};
FF_VISITABLE_STRUCT(MultiDiInput, node, idx);

struct MultiDiOutput {
  Node node;
  NodePort idx;
};
FF_VISITABLE_STRUCT(MultiDiOutput, node, idx);

MultiDiInput get_input(MultiDiEdge const &);
MultiDiOutput get_output(MultiDiEdge const &);

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

#endif
