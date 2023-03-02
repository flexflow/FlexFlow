#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "node.h"
#include "multidigraph.h"
#include "mpark/variant.hpp"

namespace FlexFlow {
namespace utils {

struct InputMultiDiEdge {
  std::size_t uid; // necessary to differentiate multiple input edges from different sources resulting from a graph cut

  utils::Node dst;
  std::size_t dstIdx;
};
bool operator==(InputMultiDiEdge const &, InputMultiDiEdge const &);

struct OutputMultiDiEdge {
  std::size_t uid; // necessary to differentiate multiple output edges from different sources resulting from a graph cut

  utils::Node src;
  std::size_t srcIdx;
};
bool operator==(OutputMultiDiEdge const &, OutputMultiDiEdge const &);

using OpenMultiDiEdge = mpark::variant<
  InputMultiDiEdge,
  OutputMultiDiEdge,
  MultiDiEdge
>;

using DownwardOpenMultiDiEdge = mpark::variant<
  OutputMultiDiEdge,
  MultiDiEdge
>;

using UpwardOpenMultDiEdge = mpark::variant<
  InputMultiDiEdge,
  MultiDiEdge
>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OutputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<utils::Node>> srcs = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs = tl::nullopt;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};

struct InputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<utils::Node>> dsts = tl::nullopt;
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
}

namespace std {

template <>
struct hash<::FlexFlow::utils::OpenMultiDiEdge> {
  size_t operator()(::FlexFlow::utils::OpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::utils::DownwardOpenMultiDiEdge> {
  size_t operator()(::FlexFlow::utils::DownwardOpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::utils::UpwardOpenMultDiEdge> {
  size_t operator()(::FlexFlow::utils::UpwardOpenMultDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::utils::OutputMultiDiEdge> {
  size_t operator()(::FlexFlow::utils::OutputMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::utils::InputMultiDiEdge> {
  size_t operator()(::FlexFlow::utils::InputMultiDiEdge const &) const;
};

}

namespace FlexFlow {
namespace utils {

struct IOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};

struct IDownwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge> query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;
};

struct IUpwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<UpwardOpenMultDiEdge> query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;
};

struct IOpenMultiDiGraph : public IOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
};

struct IUpwardOpenMultiDiGraph : public IUpwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(UpwardOpenMultDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultDiEdge const &) = 0;
};

struct IDownwardOpenMultiDiGraph : public IDownwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
};

}
}

#endif 
