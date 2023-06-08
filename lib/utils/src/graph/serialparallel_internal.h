#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_INTERNAL_H

#include "utils/graph/node.h"
#include "utils/graph/digraph.h"
#include "utils/graph/serialparallel.h"
#include "mpark/variant.hpp"
#include <vector>

namespace FlexFlow {

struct ParallelInternal;

enum class SplitType {
  SERIAL, PARALLEL
};

struct SplitASTNode;

using SplitAST = mpark::variant<SplitASTNode, Node>;

struct SplitASTNode {
  SplitASTNode(SplitType type):type(type){}
  SplitASTNode(SplitType type, SplitAST const & lhs, SplitAST const & rhs)
  :type(type) {
    children.push_back(lhs);
    children.push_back(rhs);
  }
  SplitASTNode(SplitType type, std::vector<SplitAST> const & children):type(type),children(children){}

  std::vector<SplitAST> children;
  SplitType type;
};

SplitAST sp_decomposition(DiGraphView const &g);
SplitAST parallel_decomposition(DiGraphView const &g);

std::unordered_set<Node> from_source_to_sink(DiGraphView const &, Node const &src, Node const &sink);

mpark::variant<Serial, Parallel, Node> to_final_ast(SplitAST const &);
SplitAST flatten_ast(SplitAST const &ast);

}

#endif
