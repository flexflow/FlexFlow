#include "simplification.h"
#include "spdlog/spdlog.h"
#include <queue>

namespace FlexFlow {
namespace PCG {

Simplifier::Simplifier(std::string const &logger_name)
    : logger(spdlog::get(logger_name)) {}

void Simplifier::simplify_parallel_ops() {
  logger->debug("Trying to simplify parallel ops");

  /* using FlexFlow::PCG::Utils::nodes; */
  /* using FlexFlow::PCG::Utils::predecessor; */
  /* using FlexFlow::PCG::Utils::predecessors; */
  /* using FlexFlow::PCG::Utils::successor; */

  std::queue<Node> work_queue;
  for (Node const &node : nodes(*this)) {
    if (node.ptr->is_parallel_op()) {
      work_queue.push(node);
    }
  }

  while (!work_queue.empty()) {
    Node node = work_queue.front();
    log_simplify.debug() << "Trying to simplify starting from "
                         << node.to_string();
    work_queue.pop();

    auto opt_succ = successor(*this, node);
    if (!opt_succ.has_value()) {
      log_simplify.debug() << "Skipping because does not have single successor";
      continue;
    }
    Node succ = opt_succ.value();
    if (!succ.ptr->is_parallel_op()) {
      log_simplify.debug() << "Skipping because successor is not a parallel op";
      continue;
    }

    std::vector<ParallelOpInfo> node_parallel_op_info,
        successor_parallel_op_info;
    ((ParallelOp *)node.ptr)->append_parallel_op_info(node_parallel_op_info);
    ((ParallelOp *)succ.ptr)
        ->append_parallel_op_info(successor_parallel_op_info);
    ParallelOpJoinResult result = try_join_parallel_ops(
        node_parallel_op_info.front(), successor_parallel_op_info.front());

    if (!result.join_did_succeed) {
      log_simplify.debug() << "Skipping because join did not succeed";
      continue;
    }
    log_simplify.debug() << "Did join nodes";
    log_simplify.debug() << "  " << node.to_string();
    log_simplify.debug() << "  " << succ.to_string();

    for (Node const &p : predecessors(*this, node)) {
      if (p.ptr->is_parallel_op()) {
        work_queue.push(p);
      }
    }

    Graph new_g(this->model);
    if (result.op.has_value()) {
      Node new_op = this->model->get_or_create_parallel_op_node(
          node.ptr->inputs[0], result.op.value());
      work_queue.push(new_op);
      new_g.add_node(new_op);
    }
    this->replace_subgraph({node, succ}, new_g);
  }
  log_simplify.debug() << "Finished simplifying parallel ops";
}

void Graph::simplify(SimplificationSettings const &settings) {
  // Simplify the graph by eliminating reverse parallel ops
  // and fusing multiple parallel ops
  // old graph: e1->n1->e2->n2->en
  // new graph: e1->new_node->en
  // TODO: temporarily disabled graph simplification
  if (settings.simplify_parallel_ops) {
    this->simplify_parallel_ops();
  }
  if (settings.fuse_parallel_ops) {
    bool simplify = true;
    while (simplify) {
      simplify = false;
      for (auto const &it : this->inEdges) {
        if (it.first.ptr == NULL) {
          continue;
        }
        if (it.first.ptr->is_parallel_op()) {
          Node n2 = it.first;
          assert(it.second.size() == 1);
          Edge e2 = *it.second.begin();
          Node n1 = e2.srcOp;
          // Check that n1 is a parallel op
          // Check that n1 must have a single out edge
          if (n1.ptr->is_parallel_op() &&
              this->outEdges.find(n1)->second.size() == 1) {
            // merge n1 and n2
            std::vector<ParallelOpInfo> parallel_ops;
            ((ParallelOp *)n1.ptr)->append_parallel_op_info(parallel_ops);
            ((ParallelOp *)n2.ptr)->append_parallel_op_info(parallel_ops);
            Node new_node = model->get_or_create_fused_parallel_node(
                n1.ptr->inputs[0], parallel_ops);
            auto const &inList = this->inEdges.find(n1)->second;
            assert(inList.size() == 1);
            Edge e1 = *inList.begin();
            // Update graph by adding edges
            this->add_edge(e1.srcOp, new_node, e1.srcIdx, 0);
            this->remove_edge(e1);
            this->remove_edge(e2);
            // make a copy of outList
            if (this->outEdges.find(n2) != this->outEdges.end()) {
              auto const outList = this->outEdges.find(n2)->second;
              for (auto const &e : outList) {
                this->add_edge(new_node, e.dstOp, 0, e.dstIdx);
                this->remove_edge(e);
              }
            }
            simplify = true;
          }
        }
        if (simplify) {
          break;
        }
      }
    }
  }

  if (settings.remove_trailing_parallel_ops) {
    // Remove final parallel ops
    std::vector<Node> candidates;
    for (auto const &it : this->outEdges) {
      if (it.second.size() == 0 && it.first.ptr->op_type != OP_REDUCTION &&
          it.first.ptr->op_type != OP_FUSED_PARALLEL &&
          it.first.ptr->is_parallel_op()) {
        candidates.push_back(it.first);
      }
    }
    size_t index = 0;
    while (index < candidates.size()) {
      Node parallel_op = candidates[index++];
      auto const &inList = this->inEdges.find(parallel_op)->second;
      assert(inList.size() == 1);
      Edge e = *inList.begin();
      this->remove_edge(e);
      if (this->outEdges.find(e.srcOp)->second.size() == 0 &&
          e.srcOp.ptr->is_parallel_op()) {
        candidates.push_back(e.srcOp);
      }
    }
  }

  if (settings.remove_noops) {
    // Remove NoOps
    std::vector<Node> noop_nodes;
    for (auto const &it : this->inEdges) {
      if (it.first.ptr == NULL) {
        continue;
      }
      if (it.first.ptr->op_type == OP_NOOP) {
        noop_nodes.push_back(it.first);
      }
    }
    size_t index = 0;
    while (index < noop_nodes.size()) {
      Node noop = noop_nodes[index++];
      auto const &inList = this->inEdges.find(noop)->second;
      assert(inList.size() == 1);
      Edge in_edge = *inList.begin();
      // make a copy of outList
      if (this->outEdges.find(noop) != this->outEdges.end()) {
        auto const outList = this->outEdges.find(noop)->second;
        for (auto const &e : outList) {
          this->add_edge(in_edge.srcOp, e.dstOp, in_edge.srcIdx, e.dstIdx);
          this->remove_edge(e);
        }
      }
      this->remove_edge(in_edge);
    }
  }
}

} // namespace PCG
} // namespace FlexFlow
