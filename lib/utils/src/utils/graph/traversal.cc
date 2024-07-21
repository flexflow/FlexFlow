#include "utils/graph/traversal.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/containers/contains.h"

namespace FlexFlow {

using cdi = checked_dfs_iterator;
using udi = unchecked_dfs_iterator;
using bfi = bfs_iterator;
/* using bdi = BoundaryDFSView::boundary_dfs_iterator; */

udi::unchecked_dfs_iterator(DiGraphView const &g,
                            std::vector<Node> const &stack)
    : stack(stack), graph(g) {}

udi::unchecked_dfs_iterator(DiGraphView const &g,
                            std::unordered_set<Node> const &starting_points)
    : graph(g) {
  for (Node const &n : starting_points) {
    this->stack.push_back(n);
  }
}

udi::reference udi::operator*() const {
  return this->stack.back();
}

udi::pointer udi::operator->() {
  return &this->operator*();
}

udi &udi::operator++() {
  Node const last = this->operator*();
  this->stack.pop_back();

  std::unordered_set<DirectedEdge> outgoing = get_outgoing_edges(graph, last);
  for (DirectedEdge const &e : outgoing) {
    auto it = std::find(stack.begin(), stack.end(), e.dst);
    if (it == stack.end()) {
      stack.push_back(e.dst);
    } else {
      stack.erase(it);
      stack.push_back(e.dst);
    }
  }
  return *this;
}

void udi::skip() {
  this->stack.pop_back();
}

udi udi::operator++(int) {
  auto tmp = *this;
  ++(*this);
  return tmp;
}

bool udi::operator==(udi const &other) const {
  return this->stack == other.stack;
}

bool udi::operator!=(udi const &other) const {
  return this->stack != other.stack;
}

cdi::checked_dfs_iterator(DiGraphView const &g,
                          std::vector<Node> const &stack,
                          std::unordered_set<Node> const &seen)
    : iter(g, stack), seen(seen) {}

cdi::checked_dfs_iterator(DiGraphView const &g,
                          std::unordered_set<Node> const &starting_points)
    : iter(g, starting_points), seen{} {}

cdi::reference cdi::operator*() const {
  return this->iter.operator*();
}
cdi::pointer cdi::operator->() {
  return this->iter.operator->();
}

cdi &cdi::operator++() {
  this->seen.insert(*iter);
  this->iter++;
  while (contains(this->seen, *iter)) {
    this->iter.skip();
  }
  return *this;
}

cdi cdi::operator++(int) {
  auto tmp = *this;
  ++(*this);
  return tmp;
}

bool cdi::operator==(cdi const &other) const {
  return this->iter == other.iter && this->seen == other.seen;
}

bool cdi::operator!=(cdi const &other) const {
  return this->iter != other.iter && this->seen != other.seen;
}

bfi::bfs_iterator(DiGraphView const &g,
                  std::queue<Node> const &q,
                  std::optional<std::unordered_set<Node>> const &seen)
    : graph(g), q(q), seen(seen) {}

bfi::bfs_iterator(DiGraphView const &g,
                  std::unordered_set<Node> const &starting_points)
    : graph(g), seen(std::unordered_set<Node>{}) {
  for (Node const &n : starting_points) {
    this->q.push(n);
  }
}

bfi::reference bfi::operator*() const {
  return this->q.front();
}

bfi::pointer bfi::operator->() {
  return &this->operator*();
}

bfi &bfi::operator++() {
  Node current = this->operator*();
  assert(this->seen.has_value());
  this->seen.value().insert(current);
  this->q.pop();

  std::unordered_set<DirectedEdge> outgoing =
      get_outgoing_edges(graph, {current});
  for (DirectedEdge const &e : outgoing) {
    if (!contains(this->seen.value(), e.dst)) {
      this->q.push(e.dst);
    }
  }

  while (!this->q.empty() && contains(this->seen.value(), this->q.front())) {
    this->q.pop();
  }

  return *this;
}

bfi bfi::operator++(int) {
  auto tmp = *this;
  ++(*this);
  return tmp;
}

bool bfi::operator==(bfi const &other) const {
  return this->q == other.q &&
         (!this->seen.has_value() || !other.seen.has_value() ||
          this->seen == other.seen) &&
         is_ptr_equal(this->graph, other.graph);
}

bool bfi::operator!=(bfi const &other) const {
  return this->q != other.q ||
         (this->seen.has_value() && other.seen.has_value() &&
          this->seen != other.seen) &&
             is_ptr_equal(this->graph, other.graph);
}

CheckedDFSView::CheckedDFSView(DiGraphView const &g,
                               std::unordered_set<Node> const &starting_points)
    : graph(g), starting_points(starting_points) {}

checked_dfs_iterator CheckedDFSView::cbegin() const {
  return checked_dfs_iterator(this->graph, this->starting_points);
}

checked_dfs_iterator CheckedDFSView::cend() const {
  return checked_dfs_iterator(this->graph, {}, get_nodes(this->graph));
}

checked_dfs_iterator CheckedDFSView::begin() const {
  return this->cbegin();
}

checked_dfs_iterator CheckedDFSView::end() const {
  return this->cend();
}

CheckedDFSView dfs(DiGraphView const &g,
                   std::unordered_set<Node> const &starting_points) {
  return CheckedDFSView(g, starting_points);
}

UncheckedDFSView::UncheckedDFSView(
    DiGraphView const &g, std::unordered_set<Node> const &starting_points)
    : graph(g), starting_points(starting_points) {}

unchecked_dfs_iterator UncheckedDFSView::cbegin() const {
  return unchecked_dfs_iterator(this->graph, this->starting_points);
}

unchecked_dfs_iterator UncheckedDFSView::cend() const {
  return unchecked_dfs_iterator(this->graph, std::vector<Node>{});
}

unchecked_dfs_iterator UncheckedDFSView::begin() const {
  return this->cbegin();
}

unchecked_dfs_iterator UncheckedDFSView::end() const {
  return this->cend();
}

UncheckedDFSView
    unchecked_dfs(DiGraphView const &g,
                  std::unordered_set<Node> const &starting_points) {
  return UncheckedDFSView(g, starting_points);
}

BFSView::BFSView(DiGraphView const &g,
                 std::unordered_set<Node> const &starting_points)
    : graph(g), starting_points(starting_points) {}

bfs_iterator BFSView::cbegin() const {
  return bfs_iterator(this->graph, this->starting_points);
}

bfs_iterator BFSView::cend() const {
  return bfs_iterator(this->graph, std::queue<Node>{}, std::nullopt);
}

bfs_iterator BFSView::begin() const {
  return this->cbegin();
}

bfs_iterator BFSView::end() const {
  return this->cend();
}

BFSView bfs(DiGraphView const &g,
            std::unordered_set<Node> const &starting_points) {
  return BFSView(g, starting_points);
}

} // namespace FlexFlow
