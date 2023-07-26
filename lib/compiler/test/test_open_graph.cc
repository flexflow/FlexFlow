#include "compiler/unity_algorithm.h"
#include "doctest.h"

using namespace FlexFlow;

TEST_CASE("get_source_sink_open_graph:basic") {
  OpenMultiDiGraph g(LabelledOpenMultiDiGraph<int, int>::create<
                     UnorderedLabelledOpenMultiDiGraph<int, int>>());

  int s0 = 100000;

  Node n0 = g.add_node();

  g.add_edge(InputMultiDiEdge({s0, n0.value()}, n0, 0));

  CHECK(get_closed_sources(g) == std::unordered_set<Node>{});
  CHECK(get_closed_sinks(g) == std::unordered_set<Node>{n0});

  CHECK(get_open_sources(g) == std::unordered_set<Node>{n0});
  CHECK(get_open_sinks(g) == std::unordered_set<Node>{});
}

TEST_CASE("get_source_sink_open_graph:unconnected") {
  OpenMultiDiGraph g(LabelledOpenMultiDiGraph<int, int>::create<
                     UnorderedLabelledOpenMultiDiGraph<int, int>>());
  int s0 = 100000;
  int t0 = s0 + 1;

  Node n0 = g.add_node();
  Node n1 = g.add_node();

  g.add_edge(InputMultiDiEdge({s0, n0.value()}, n0, 0));
  g.add_edge(OutputMultiDiEdge({n1.value(), t0}, n1, 0));

  /*
    g:  ->n0
        n1->
  */

  CHECK(get_closed_sources(g) == std::unordered_set<Node>{n1});
  CHECK(get_closed_sinks(g) == std::unordered_set<Node>{n0});

  CHECK(get_open_sources(g) == std::unordered_set<Node>{n0});
  CHECK(get_open_sinks(g) == std::unordered_set<Node>{n1});
}

TEST_CASE("get_source_sink_open_graph:complex") {
  OpenMultiDiGraph g(LabelledOpenMultiDiGraph<int, int>::create<
                     UnorderedLabelledOpenMultiDiGraph<int, int>>());
  int s0 = 100000;
  int s1 = s0 + 1;
  int t0 = s1 + 1;
  int t1 = t0 + 1;

  std::vector<Node> ns;
  for (int i = 0; i < 8; ++i) {
    ns.push_back(g.add_node());
  }

  g.add_edge(InputMultiDiEdge({s0, ns[0].value()}, ns[0], 0));
  g.add_edge(MultiDiEdge(ns[0], ns[1], 0, 0));
  g.add_edge(OutputMultiDiEdge({ns[1].value(), t0}, ns[1], 0));
  g.add_edge(OutputMultiDiEdge({ns[1].value(), t1}, ns[1], 1));

  g.add_edge(MultiDiEdge(ns[2], ns[3], 0, 0));
  g.add_edge(MultiDiEdge(ns[2], ns[4], 1, 0));
  g.add_edge(MultiDiEdge(ns[4], ns[3], 0, 1));
  g.add_edge(OutputMultiDiEdge({ns[3].value(), t1}, ns[3], 0));

  g.add_edge(InputMultiDiEdge({s0, ns[5].value()}, ns[5], 0));
  g.add_edge(InputMultiDiEdge({s1, ns[5].value()}, ns[5], 1));
  g.add_edge(MultiDiEdge(ns[5], ns[6], 0, 0));
  g.add_edge(MultiDiEdge(ns[6], ns[7], 0, 0));

  CHECK(get_closed_sources(g) == std::unordered_set<Node>{ns[2]});
  CHECK(get_closed_sinks(g) == std::unordered_set<Node>{ns[7]});

  CHECK(get_open_sources(g) == std::unordered_set<Node>{ns[1], ns[5]});
  CHECK(get_open_sinks(g) == std::unordered_set<Node>{ns[1], ns[3]});
}

TEST_CASE("get_cut") {
  auto g = LabelledOpenMultiDiGraph<int, int>::create<
      UnorderedLabelledOpenMultiDiGraph<int, int>>;

  std::vector<Node> ns = add_nodes(g, 5);

  int t0 = 100000;

  MultiDiEdge e0(ns[0], ns[1], 0, 0);
  MultiDiEdge e1(ns[1], ns[2], 0, 0);
  MultiDiEdge e2(ns[1], ns[3], 1, 0);
  MultiDiEdge e3(ns[2], ns[4], 0, 0);
  MultiDiEdge e4(ns[3], ns[4], 0, 1);
  OutputMultiDiEdge e5({ns[4].value(), t0}, ns[4], 0);

  GraphSplit gs0{{ns[0], ns[1]}, {ns[2], ns[3], ns[4]}};
  CHECK(get_cut(g, gs0) == std::unordered_set<MultiDiEdge>{e1, e2});

  GraphSplit gs1{{ns[0], ns[1], ns[2], ns[3]}, {ns[4]}};
  CHECK(get_cut(g, gs1) == std::unordered_set<MultiDiEdge>{e3, e4});
}
