#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_H

#include "utils/visitable.h"
#include "tensor.h"
#include "parallel_tensor.h"
#include "operator_attrs.h"
#include "utils/json.h"
#include "utils/required.h"

namespace FlexFlow {

struct V1GraphOutput : public use_visitable_cmp<V1GraphOutput> {
  req<size_t> srcNode;
  req<size_t> srcIdx;
};
FF_VISITABLE_STRUCT(V1GraphOutput, srcNode, srcIdx);

struct V1GraphEdge : public use_visitable_cmp<V1GraphEdge> {
  req<size_t> srcNode;
  req<size_t> srcIdx;
  req<size_t> dstNode;
  req<size_t> dstIdx;
};
FF_VISITABLE_STRUCT(V1GraphEdge, srcNode, srcIdx, dstNode, dstIdx);

template <typename NodeT, typename TensorT> 
struct V1JsonableGraph : public use_visitable_cmp<V1JsonableGraph<NodeT, TensorT>> {
  req<std::unordered_set<size_t, NodeT>> nodes;
  req<int> edges;
  req<std::unordered_map<NodeT, TensorT>> tensors;
};

struct V1Layer : public use_visitable_cmp<V1Layer> {
  V1CompGraphOperatorAttrs attrs;
  req<optional<std::string>> name;
};
FF_VISITABLE_STRUCT(V1Layer);

using V1ComputationGraph = V1JsonableGraph<V1Layer, V1Tensor>;
FF_VISITABLE_STRUCT(V1ComputationGraph, nodes, edges, tensors);
using V1ParallelComputationGraph = V1JsonableGraph<V1PCGOperatorAttrs, V1ParallelTensor>;
FF_VISITABLE_STRUCT(V1ParallelComputationGraph, nodes, edges, tensors);

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
