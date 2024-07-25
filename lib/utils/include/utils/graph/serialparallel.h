#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "digraph.h"
#include "multidigraph.h"
#include "utils/optional.h"
#include <variant>
#include <vector>

namespace FlexFlow {
bool has_single_source(DiGraphView const &g);
bool has_single_sink(DiGraphView const &g);

Node find_source_node(DiGraphView const &);
Node find_sink_node(DiGraphView const &);

std::optional<Node> find_bottleneck_node(DiGraphView const &);

struct Parallel;

struct Serial {
  std::vector<std::variant<Parallel, Node>> children;
};

struct Parallel {
  std::vector<std::variant<Serial, Node>> children;
  bool operator==(Parallel const &other) const;
  bool operator!=(Parallel const &other) const;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Parallel, children);
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Serial, children);

using SerialParallelDecomposition = std::variant<Serial, Parallel, Node>;

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);

std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
                                               MultiDiGraph const &ext);

std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
                                             MultiDiGraph const &ext);

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext);

std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext);

MultiDiGraph serial_composition(MultiDiGraph const &g1, MultiDiGraph const &g2);

MultiDiGraph parallel_composition(MultiDiGraph const &g1,
                                  MultiDiGraph const &g2);

SerialParallelDecomposition parallel_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions);

SerialParallelDecomposition serial_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions);

MultiDiGraph multidigraph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition);

bool isempty(SerialParallelDecomposition const &sp);

SerialParallelDecomposition normalize(SerialParallelDecomposition const &sp);

std::unordered_map<Node, size_t>
    node_counter(SerialParallelDecomposition const &sp);

size_t node_count(SerialParallelDecomposition const &sp);

std::vector<SerialParallelDecomposition>
    to_sp_decomp(std::vector<std::variant<Parallel, Node>> const &children);

std::vector<SerialParallelDecomposition>
    to_sp_decomp(std::vector<std::variant<Serial, Node>> const &children);

SerialParallelDecomposition
    to_sp_decomp(std::variant<Parallel, Node> const &child);

SerialParallelDecomposition
    to_sp_decomp(std::variant<Serial, Node> const &child);

size_t node_count(DiGraphView const &g);

float compute_cost(SerialParallelDecomposition const &sp,
                   std::unordered_map<Node, float> cost_map);

float compute_cost(DiGraphView const &g,
                   std::unordered_map<Node, float> const &cost_map);

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map);

float critical_path_cost(SerialParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map);

float average_parallelism_degree(
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map);

float max_parallelism_degree(SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map);

float relative_cost_increase(DiGraphView const &g,
                             SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map);

float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map);

} // namespace FlexFlow

#endif
