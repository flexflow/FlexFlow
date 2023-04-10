#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "node.h"
#include "multidigraph.h"
#include "utils/variant.h"
#include "tl/optional.hpp"
#include "utils/visitable.h"

namespace FlexFlow {

struct InputMultiDiEdge {
  std::pair<std::size_t, std::size_t> uid; // necessary to differentiate multiple input edges from different sources resulting from a graph cut

  Node dst;
  std::size_t dstIdx;
};
bool operator==(InputMultiDiEdge const &, InputMultiDiEdge const &);

struct OutputMultiDiEdge {
  std::pair<std::size_t, std::size_t> uid; // necessary to differentiate multiple output edges from different sources resulting from a graph cut

  Node src;
  std::size_t srcIdx;
};
bool operator==(OutputMultiDiEdge const &, OutputMultiDiEdge const &);

using OpenMultiDiEdge = variant<
  InputMultiDiEdge,
  OutputMultiDiEdge,
  MultiDiEdge
>;

using DownwardOpenMultiDiEdge = variant<
  OutputMultiDiEdge,
  MultiDiEdge
>;

using UpwardOpenMultiDiEdge = variant<
  InputMultiDiEdge,
  MultiDiEdge
>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OutputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs = tl::nullopt;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};

struct InputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> dsts = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> dstIdxs = tl::nullopt;

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

}

namespace std {

template <>
struct hash<::FlexFlow::OpenMultiDiEdge> {
  size_t operator()(::FlexFlow::OpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::DownwardOpenMultiDiEdge> {
  size_t operator()(::FlexFlow::DownwardOpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::UpwardOpenMultiDiEdge> {
  size_t operator()(::FlexFlow::UpwardOpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::OutputMultiDiEdge> {
  size_t operator()(::FlexFlow::OutputMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::InputMultiDiEdge> {
  size_t operator()(::FlexFlow::InputMultiDiEdge const &) const;
};

}

VISITABLE_STRUCT(::FlexFlow::InputMultiDiEdge, uid, dst, dstIdx);
VISITABLE_STRUCT(::FlexFlow::OutputMultiDiEdge, uid, src, srcIdx);

namespace FlexFlow {

struct IOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};

struct IDownwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge> query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;
};

struct IUpwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge> query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;
};

struct IOpenMultiDiGraph : public IOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
};

struct IUpwardOpenMultiDiGraph : public IUpwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
};

struct IDownwardOpenMultiDiGraph : public IDownwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
};

}

#endif 
