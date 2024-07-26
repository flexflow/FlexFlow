#include "utils/graph/serial_parallel/series_reduction.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_pre/post/center_node") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    std::vector<MultiDiEdge> e = add_edges(g,
                                           {
                                               {n.at(0), n.at(1)},
                                               {n.at(1), n.at(2)},
                                           });
    SeriesReduction reduction = make_series_reduction(e.at(0), e.at(1));

    SUBCASE("get_pre_node") {
      Node result = get_pre_node(g, reduction);
      Node correct = n.at(0);
      CHECK(result == correct);
    }

    SUBCASE("get_post_node") {
      Node result = get_post_node(g, reduction);
      Node correct = n.at(2);
      CHECK(result == correct);
    }

    SUBCASE("get_center_node") {
      Node result = get_center_node(g, reduction);
      Node correct = n.at(1);
      CHECK(result == correct);
    }
  }

  TEST_CASE("find_series_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 3);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(1), n.at(2)},
                                             });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::optional<SeriesReduction> correct =
          make_series_reduction(e.at(0), e.at(1));
      CHECK(result == correct);
    }

    SUBCASE("does not find if other edges are involved with center node") {
      SUBCASE("duplicate edge") {
        std::vector<Node> n = add_nodes(g, 3);
        std::vector<MultiDiEdge> e = add_edges(g,
                                               {
                                                   {n.at(0), n.at(1)},
                                                   {n.at(0), n.at(1)},
                                                   {n.at(1), n.at(2)},
                                               });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("misc edge") {
        std::vector<Node> n = add_nodes(g, 4);
        std::vector<MultiDiEdge> e = add_edges(g,
                                               {
                                                   {n.at(0), n.at(1)},
                                                   {n.at(1), n.at(3)},
                                                   {n.at(1), n.at(2)},
                                               });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct = std::nullopt;
        CHECK(result == correct);
      }
    }

    SUBCASE("does find if other edges are involved with non-center node") {
      std::vector<Node> n = add_nodes(g, 4);
      SUBCASE("edge from dst") {
        std::vector<MultiDiEdge> e = add_edges(g,
                                               {
                                                   {n.at(0), n.at(1)},
                                                   {n.at(1), n.at(2)},

                                                   {n.at(2), n.at(3)},
                                                   {n.at(3), n.at(2)},
                                                   {n.at(0), n.at(3)},
                                                   {n.at(3), n.at(0)},
                                               });

        std::optional<SeriesReduction> result = find_series_reduction(g);
        std::optional<SeriesReduction> correct =
            make_series_reduction(e.at(0), e.at(1));
        CHECK(result == correct);
      }
    }

    SUBCASE("finds one reduction when there are multiple") {
      std::vector<Node> n = add_nodes(g, 4);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(1), n.at(2)},
                                                 {n.at(2), n.at(3)},
                                             });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::unordered_set<SeriesReduction> correct_options = {
          make_series_reduction(e.at(0), e.at(1)),
          make_series_reduction(e.at(1), e.at(2)),
      };
      CHECK(result.has_value());
      CHECK(contains(correct_options, result.value()));
    }

    SUBCASE("in larger graph") {
      std::vector<Node> n = add_nodes(g, 8);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(2)},
                                                 {n.at(1), n.at(2)},
                                                 {n.at(2), n.at(3)},
                                                 {n.at(2), n.at(3)},
                                                 {n.at(3), n.at(4)}, //*
                                                 {n.at(4), n.at(5)}, //*
                                                 {n.at(5), n.at(6)},
                                                 {n.at(5), n.at(7)},
                                             });

      std::optional<SeriesReduction> result = find_series_reduction(g);
      std::optional<SeriesReduction> correct =
          make_series_reduction(e.at(4), e.at(5));
      CHECK(result == correct);
    }
  }

  TEST_CASE("apply_series_reduction") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 3);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(1)},
                                                 {n.at(1), n.at(2)},
                                             });

      SeriesReduction reduction = make_series_reduction(e.at(0), e.at(1));

      MultiDiEdge returned_edge = apply_series_reduction(g, reduction);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(g);
        std::unordered_set<Node> correct_nodes = {n.at(0), n.at(2)};
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edges") {
        std::unordered_set<MultiDiEdge> result_edges = get_edges(g);
        std::unordered_set<MultiDiEdge> correct_edges = {returned_edge};
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("returned edge") {
        SUBCASE("src") {
          Node returned_edge_src = g.get_multidiedge_src(returned_edge);
          Node correct_src = n.at(0);
          CHECK(returned_edge_src == correct_src);
        }

        SUBCASE("dst") {
          Node returned_edge_dst = g.get_multidiedge_dst(returned_edge);
          Node correct_dst = n.at(2);
          CHECK(returned_edge_dst == correct_dst);
        }
      }
    }

    SUBCASE("in larger graph") {
      std::vector<Node> n = add_nodes(g, 8);
      std::vector<MultiDiEdge> e = add_edges(g,
                                             {
                                                 {n.at(0), n.at(2)},
                                                 {n.at(1), n.at(2)},
                                                 {n.at(2), n.at(3)},
                                                 {n.at(2), n.at(3)},
                                                 {n.at(3), n.at(4)}, //*
                                                 {n.at(4), n.at(5)}, //*
                                                 {n.at(5), n.at(6)},
                                                 {n.at(5), n.at(7)},
                                             });

      MultiDiEdge reduction_e1 = e.at(4);
      MultiDiEdge reduction_e2 = e.at(5);
      SeriesReduction reduction =
          make_series_reduction(reduction_e1, reduction_e2);

      MultiDiEdge returned_edge = apply_series_reduction(g, reduction);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(g);
        std::unordered_set<Node> correct_nodes =
            set_minus(unordered_set_of(n), {n.at(4)});
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edges") {
        std::unordered_set<MultiDiEdge> result_edges = get_edges(g);
        std::unordered_set<MultiDiEdge> correct_edges = [&] {
          std::unordered_set<MultiDiEdge> new_edges = unordered_set_of(e);
          new_edges.erase(reduction_e1);
          new_edges.erase(reduction_e2);
          new_edges.insert(returned_edge);
          return new_edges;
        }();
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("returned edge") {
        SUBCASE("src") {
          Node returned_edge_src = g.get_multidiedge_src(returned_edge);
          Node correct_src = n.at(3);
          CHECK(returned_edge_src == correct_src);
        }

        SUBCASE("dst") {
          Node returned_edge_dst = g.get_multidiedge_dst(returned_edge);
          Node correct_dst = n.at(5);
          CHECK(returned_edge_dst == correct_dst);
        }
      }
    }
  }
}
