#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "test/utils/doctest.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("pattern_split (sequential)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n0_added = g.add_node({}, 1);
    Node n0 = n0_added.node;
    OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

    NodeAddedResult n1_added = g.add_node({v0}, 1);
    Node n1 = n1_added.node;
    OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

    UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};
    PatternNode p0 = PatternNode{n0};
    PatternNode p1 = PatternNode{n1};
    PatternValue pv0 = pattern_value_from_raw_open_dataflow_value(v0);
    PatternValue pv1 = pattern_value_from_raw_open_dataflow_value(v1);

    PatternSplit even_split = PatternSplit{
        std::unordered_set<PatternNode>{p0},
        std::unordered_set<PatternNode>{p1},
    };

    SUBCASE("find_even_split") {
      PatternSplit result = find_even_split(pattern);
      PatternSplit correct = even_split;
      CHECK(result == correct);
    }

    SUBCASE("apply_split") {
      PatternSplitResult split_result = apply_split(pattern, even_split);
      SUBCASE("subpattern_1") {
        std::unordered_set<PatternNode> result =
            get_nodes(split_result.subpattern_1);
        std::unordered_set<PatternNode> correct = even_split.first;
        CHECK(result == correct);
      }
      SUBCASE("subpattern_2") {
        std::unordered_set<PatternNode> result =
            get_nodes(split_result.subpattern_2);
        std::unordered_set<PatternNode> correct = even_split.second;
        CHECK(result == correct);
      }
      SUBCASE("full_pattern_values_to_subpattern_1_inputs") {
        bidict<PatternValue, PatternInput> result =
            split_result.full_pattern_values_to_subpattern_1_inputs;
        bidict<PatternValue, PatternInput> correct = {};
        CHECK(result == correct);
      }
      SUBCASE("full_pattern_values_to_subpattern_2_inputs") {
        bidict<PatternValue, PatternInput> result =
            split_result.full_pattern_values_to_subpattern_2_inputs;
        PatternInput i0 = get_only(get_inputs(split_result.subpattern_2));
        bidict<PatternValue, PatternInput> correct = {
            {pv0, i0},
        };
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("pattern split (parallel)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = g.add_input();
    DataflowGraphInput i1 = g.add_input();

    NodeAddedResult n0_added = g.add_node({OpenDataflowValue{i0}}, 1);
    Node n0 = n0_added.node;
    OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

    NodeAddedResult n1_added = g.add_node({OpenDataflowValue{i1}}, 1);
    Node n1 = n1_added.node;
    OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

    UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};
    PatternInput pi0 = PatternInput{i0};
    PatternInput pi1 = PatternInput{i1};
    PatternNode p0 = PatternNode{n0};
    PatternNode p1 = PatternNode{n1};
    PatternValue pv0 = pattern_value_from_raw_open_dataflow_value(v0);
    PatternValue pv1 = pattern_value_from_raw_open_dataflow_value(v1);

    PatternSplit even_split = PatternSplit{
        std::unordered_set<PatternNode>{p0},
        std::unordered_set<PatternNode>{p1},
    };

    SUBCASE("apply_split") {
      PatternSplitResult split_result = apply_split(pattern, even_split);
      SUBCASE("subpattern_1") {
        std::unordered_set<PatternNode> result =
            get_nodes(split_result.subpattern_1);
        std::unordered_set<PatternNode> correct = even_split.first;
        CHECK(result == correct);
      }
      SUBCASE("subpattern_2") {
        std::unordered_set<PatternNode> result =
            get_nodes(split_result.subpattern_2);
        std::unordered_set<PatternNode> correct = even_split.second;
        CHECK(result == correct);
      }
      SUBCASE("full_pattern_values_to_subpattern_1_inputs") {
        bidict<PatternValue, PatternInput> result =
            split_result.full_pattern_values_to_subpattern_1_inputs;
        bidict<PatternValue, PatternInput> correct = {
            {PatternValue{pi0},
             get_only(get_inputs(split_result.subpattern_1))},
        };
        CHECK(result == correct);
      }
      SUBCASE("full_pattern_values_to_subpattern_2_inputs") {
        bidict<PatternValue, PatternInput> result =
            split_result.full_pattern_values_to_subpattern_2_inputs;
        bidict<PatternValue, PatternInput> correct = {
            {PatternValue{pi1},
             get_only(get_inputs(split_result.subpattern_2))},
        };
        CHECK(result == correct);
      }
    }
  }
}
