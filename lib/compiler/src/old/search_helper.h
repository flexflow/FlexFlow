#ifndef _FLEXFLOW_FFC_SRC_SEARCH_HELPER_H
#define _FLEXFLOW_FFC_SRC_SEARCH_HELPER_H

#include "graph.h"
#include "split_types.h"

namespace FlexFlow {

struct GraphCostResult {
  float cost;
  std::unordered_map<Node, MachineView> views;

  static GraphCostResult invalid();

  bool operator<(GraphCostResult const &other) const;

  friend std::ostream &operator<<(std::ostream &, GraphCostResult const &);
};

template <typename T>
T sequence_cost(T const &first, T const &second);

template <typename T>
T parallel_cost(T const &first, T const &second);

class SearchHelper {
public:
  SearchHelper();

  template <typename T>
  T graph_cost(Graph const *graph,
               NodeAssignment const &source,
               NodeAssignment const &sink,
               MachineResource const &resources,
               bool include_sink_compute_time) const;
  template <typename T>
  T find_optimal_sequence_graph_time(Graph const *g,
                                     Node const &bottleneck_node,
                                     NodeAssignment const &source,
                                     NodeAssignment const &sink,
                                     MachineResource const &resources) const;
  template <typename T>
  T find_optimal_nonsequence_graph_time(Graph const *g,
                                        NodeAssignment const &source,
                                        NodeAssignment const &sink,
                                        MachineResource const &resources) const;
  /* void find_optimal_nonsequence_graph_views(Graph const *g, */
  /*                                           NodeAssignment const &source, */
  /*                                           NodeAssignment const &sink, */
  /*                                           MachineResource const &resources,
   */
  /*                                           float optimal_cost, */
  /*                                           std::unordered_map<Node,
   * MachineView>& optimal_views) const; */
  std::vector<MachineView>
      get_valid_machine_views(Node const &node,
                              MachineResource const &resource,
                              bool log = false) const;
  std::vector<MachineView>
      get_valid_machine_views(PCGOperatorAttrs const &op,
                              MachineResource const &resource,
                              bool log = false) const;

  template <typename T>
  std::pair<bool, T> try_get_cost_from_cache(size_t hash) const;

  template <typename T>
  void try_cache_result(size_t hash, T const &value) const;

  template <typename T>
  T infinity() const;

  template <typename T>
  T empty() const;

  template <typename T>
  bool is_invalid(T const &) const;

  template <typename T>
  T estimate_xfer_cost(Graph const *g,
                       NodeAssignment const &source,
                       NodeAssignment const &sink) const;

  template <typename T>
  void add_operator_cost(NodeAssignment const &, float, T *) const;

  template <typename T>
  float get_cost(T const &) const;

  template <typename T>
  void check_matches_graph(Graph const *, T const &, Node const &) const;

public:
  mutable std::unique_ptr<RecursiveLogger> logger;

private:
  template <typename T>
  T execute_nonsequence_split(std::unique_ptr<Graph> const &first_graph,
                              std::unique_ptr<Graph> const &second_graph,
                              NodeAssignment const &source,
                              NodeAssignment const &sink,
                              MachineResource const &resources,
                              NonsequenceSplit const &split) const;

  template <typename T>
  T execute_sequence_split(std::unique_ptr<Graph> const &first_graph,
                           std::unique_ptr<Graph> const &second_graph,
                           NodeAssignment const &source,
                           NodeAssignment const &sink,
                           MachineResource const &resources,
                           SequenceSplit const &split) const;

private:
  mutable std::unordered_map<size_t, float> cached_graph_costs;
  mutable std::unordered_map<size_t,
                             std::unique_ptr<const std::vector<MachineView>>>
      cached_operator_valid_views;
};

} // namespace FlexFlow

#endif
