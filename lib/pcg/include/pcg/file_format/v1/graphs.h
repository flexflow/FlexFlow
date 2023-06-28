#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H

#include "utils/visitable.h"
#include "tensor.h"
#include "parallel_tensor.h"
#include "operator_attrs.h"
#include "utils/json.h"
#include "utils/required.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph.h"

namespace FlexFlow {

struct V1GraphOutput {
  req<size_t> srcNode;
  req<size_t> srcIdx;
};
FF_VISITABLE_STRUCT(V1GraphOutput, srcNode, srcIdx);

struct V1GraphEdge {
  req<size_t> srcNode;
  req<size_t> srcIdx;
  req<size_t> dstNode;
  req<size_t> dstIdx;
};
FF_VISITABLE_STRUCT(V1GraphEdge, srcNode, srcIdx, dstNode, dstIdx);

template <typename NodeT, typename TensorT> 
struct V1JsonableGraph {
  using node_id = size_t;
  using tensor_id = size_t;

  req<std::vector<NodeT>> nodes;
  req<std::vector<TensorT>> tensors;
  req<std::set<std::tuple<tensor_id, node_id>>> edges;
};

struct V1Layer {
  V1CompGraphOperatorAttrs attrs;
  req<optional<std::string>> name;
};
FF_VISITABLE_STRUCT(V1Layer, attrs, name);
V1Layer to_v1(Layer const &);

using V1ComputationGraph = V1JsonableGraph<V1Layer, V1Tensor>;
FF_VISITABLE_STRUCT(V1ComputationGraph, nodes, tensors, edges);
V1ComputationGraph to_v1(ComputationGraph const &);

using V1ParallelComputationGraph = V1JsonableGraph<V1PCGOperatorAttrs, V1ParallelTensor>;
FF_VISITABLE_STRUCT(V1ParallelComputationGraph, nodes, tensors, edges);
V1ParallelComputationGraph to_v1(ParallelComputationGraph const &);

}

namespace FlexFlow {
static_assert(std::is_move_constructible<V1GraphOutput>::value, "");
static_assert(is_visitable<V1GraphOutput>::value, "");
static_assert(negation<std::is_default_constructible<V1GraphOutput>>::value, "");
static_assert(elements_satisfy<is_json_serializable, V1GraphOutput>::value, "");
static_assert(is_jsonable<V1GraphOutput>::value, "");
static_assert(is_jsonable<V1GraphEdge>::value, "");
static_assert(is_jsonable<V1ComputationGraph>::value, "");
static_assert(is_jsonable<V1ParallelComputationGraph>::value, "");
}


#endif
