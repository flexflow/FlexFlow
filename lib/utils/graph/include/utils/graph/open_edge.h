#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_OPEN_EDGE
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_OPEN_EDGE

#include "multidiedge.h"

namespace FlexFlow {

using OpenMultiDiEdge =
    variant<InputMultiDiEdge, OutputMultiDiEdge, MultiDiEdge>;

using DownwardOpenMultiDiEdge = variant<OutputMultiDiEdge, MultiDiEdge>;

using UpwardOpenMultiDiEdge = variant<InputMultiDiEdge, MultiDiEdge>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OpenMultiDiEdgeQuery {
  OpenMultiDiEdgeQuery() = delete;
  OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &input_edge_query,
                       MultiDiEdgeQuery const &standard_edge_query,
                       OutputMultiDiEdgeQuery const &output_edge_query);

  OpenMultiDiEdgeQuery(MultiDiEdgeQuery const &q);
  OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &q);
  OpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &q);

  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
  OutputMultiDiEdgeQuery output_edge_query;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(OpenMultiDiEdgeQuery,
                    input_edge_query,
                    standard_edge_query,
                    output_edge_query);

struct DownwardOpenMultiDiEdgeQuery {
  DownwardOpenMultiDiEdgeQuery() = delete;
  DownwardOpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &output_edge_query,
                               MultiDiEdgeQuery const &standard_edge_query);
  DownwardOpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &output_edge_query);
  DownwardOpenMultiDiEdgeQuery(MultiDiEdgeQuery const &standard_edge_query);

  operator OpenMultiDiEdgeQuery() const;

  OutputMultiDiEdgeQuery output_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(DownwardOpenMultiDiEdgeQuery,
                                             output_edge_query,
                                             standard_edge_query);

struct UpwardOpenMultiDiEdgeQuery {
  UpwardOpenMultiDiEdgeQuery() = delete;
  UpwardOpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &,
                             MultiDiEdgeQuery const &);
  UpwardOpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &);
  UpwardOpenMultiDiEdgeQuery(MultiDiEdgeQuery const &);
  operator OpenMultiDiEdgeQuery() const;

  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(UpwardOpenMultiDiEdgeQuery,
                                             input_edge_query,
                                             standard_edge_query);

}

#endif
