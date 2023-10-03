#include "utils/graph/open_edge.h"

namespace FlexFlow {

bool is_input_edge(OpenMultiDiEdge const &e) {
  return holds_alternative<InputMultiDiEdge>(e);
}

bool is_output_edge(OpenMultiDiEdge const &e) {
  return holds_alternative<MultiDiEdge>(e);
}

bool is_standard_edge(OpenMultiDiEdge const &e) {
  return holds_alternative<OutputMultiDiEdge>(e);
}

OpenMultiDiEdgeQuery::OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &input_edge_query,
                      MultiDiEdgeQuery const &standard_edge_query,
                      OutputMultiDiEdgeQuery const &output_edge_query);

OpenMultiDiEdgeQuery::OpenMultiDiEdgeQuery(MultiDiEdgeQuery const &q)
    : OpenMultiDiEdgeQuery(
          InputMultiDiEdgeQuery::none(), q, OutputMultiDiEdgeQuery::none()) {}
OpenMultiDiEdgeQuery::OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &q)
    : OpenMultiDiEdgeQuery(
          q, MultiDiEdgeQuery::none(), OutputMultiDiEdgeQuery::none()) {}
OpenMultiDiEdgeQuery::OpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &q)
    : OpenMultiDiEdgeQuery(
          InputMultiDiEdgeQuery::none(), MultiDiEdgeQuery::none(), q) {}

DownwardOpenMultiDiEdgeQuery::DownwardOpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &output_edge_query,
                              MultiDiEdgeQuery const &standard_edge_query)
    : output_edge_query(output_edge_query),
      standard_edge_query(standard_edge_query) {}
DownwardOpenMultiDiEdgeQuery::DownwardOpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery const &output_edge_query)
    : DownwardOpenMultiDiEdgeQuery(output_edge_query,
                                    MultiDiEdgeQuery::none()) {}
DownwardOpenMultiDiEdgeQuery::DownwardOpenMultiDiEdgeQuery(MultiDiEdgeQuery const &standard_edge_query)
    : DownwardOpenMultiDiEdgeQuery(OutputMultiDiEdgeQuery::all(),
                                    standard_edge_query){};

DownwardOpenMultiDiEdgeQuery::operator OpenMultiDiEdgeQuery() const {
  return OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery::none(), standard_edge_query, output_edge_query);
}

UpwardOpenMultiDiEdgeQuery::UpwardOpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &input_edge_query,
                             MultiDiEdgeQuery const &standard_edge_query) : input_edge_query(input_edge_query), standard_edge_query(standard_edge_query) {}

UpwardOpenMultiDiEdgeQuery::UpwardOpenMultiDiEdgeQuery(InputMultiDiEdgeQuery const &input_edge_query)
  : input_edge_query(input_edge_query), standard_edge_query(MultiDiEdgeQuery::none()) {}
UpwardOpenMultiDiEdgeQuery::UpwardOpenMultiDiEdgeQuery(MultiDiEdgeQuery const &standard_edge_query)
  : input_edge_query(InputMultiDiEdgeQuery::none()), standard_edge_query(standard_edge_query) {}

}