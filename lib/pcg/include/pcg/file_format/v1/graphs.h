#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H

#include "operator_attrs.h"
#include "parallel_tensor.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph.h"
#include "tensor.h"
#include "utils/json.h"
#include "utils/required.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1GraphOutput {
  req<size_t> srcNode;
  req<size_t> srcIdx;
};
FF_VISITABLE_STRUCT(V1GraphOutput, srcNode, srcIdx);
CHECK_IS_JSONABLE(V1GraphOutput);

struct V1GraphEdge {
  req<size_t> srcNode;
  req<size_t> srcIdx;
  req<size_t> dstNode;
  req<size_t> dstIdx;
};
FF_VISITABLE_STRUCT(V1GraphEdge, srcNode, srcIdx, dstNode, dstIdx);
CHECK_IS_JSONABLE(V1GraphEdge);

struct V1MultiDiGraph {
  req<std::vector<size_t>> nodes;
  req<std::vector<size_t>> ports;
  req<std::unordered_set<V1GraphEdge>> edges;
};
FF_VISITABLE_STRUCT(V1MultiDiGraph, nodes, ports, edges);
CHECK_IS_JSONABLE(V1MultiDiGraph);
V1MultiDiGraph to_v1(MultiDiGraphView const &);
V1MultiDiGraph to_v1(MultiDiGraphView const &,
                     std::unordered_map<Node, size_t> const &,
                     std::unordered_map<NodePort, size_t> const &);

template <typename NodeT, typename TensorT>
struct V1JsonableGraph {
  using node_id = size_t;
  using tensor_id = size_t;

  req<std::unordered_map<size_t, NodeT>> node_labels;
  req<std::unordered_map<size_t, V1GraphOutput>> outputs;
  req<std::unordered_map<size_t, TensorT>> output_labels;
  V1MultiDiGraph graph;
};

struct V1Layer {
  V1CompGraphOperatorAttrs attrs;
  req<optional<std::string>> name;
};
FF_VISITABLE_STRUCT(V1Layer, attrs, name);
V1Layer to_v1(Layer const &);

using V1ComputationGraph = V1JsonableGraph<V1Layer, V1Tensor>;
FF_VISITABLE_STRUCT(
    V1ComputationGraph, node_labels, outputs, output_labels, graph);
CHECK_IS_JSONABLE(V1ComputationGraph);
V1ComputationGraph to_v1(ComputationGraph const &);

using V1ParallelComputationGraph =
    V1JsonableGraph<V1PCGOperatorAttrs, V1ParallelTensor>;
FF_VISITABLE_STRUCT(
    V1ParallelComputationGraph, node_labels, outputs, output_labels, graph);
CHECK_IS_JSONABLE(V1ParallelComputationGraph);
V1ParallelComputationGraph to_v1(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
