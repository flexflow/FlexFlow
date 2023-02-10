#include "utils/graph/traversal.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {
namespace utils {

using cdi = checked_dfs_iterator;
using udi = unchecked_dfs_iterator;
/* using bdi = BoundaryDFSView::boundary_dfs_iterator; */

udi::unchecked_dfs_iterator(IDiGraphView const &g, std::vector<Node> const &stack)
  : stack(stack), graph(&g)
{ }

udi::unchecked_dfs_iterator(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) 
  : graph(&g)
{
  for (Node const &n : starting_points) {
    this->stack.push_back(n);
  }
}

udi::reference udi::operator*() const { return this->stack.back(); }

udi::pointer udi::operator->() { return &this->operator*(); }

udi& udi::operator++() {
  Node const last = this->operator*();
  this->stack.pop_back();

  std::unordered_set<DirectedEdge> outgoing = get_outgoing_edges(*graph, {last});
  for (DirectedEdge const &e : outgoing) {
    stack.push_back(e.dst);
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

cdi::checked_dfs_iterator(IDiGraphView const &g, std::vector<Node> const &stack, std::unordered_set<Node> const &seen) 
  : iter(g, stack), seen(seen)
{ }

cdi::checked_dfs_iterator(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) 
  : iter(g, starting_points), seen{}
{ }

cdi::reference cdi::operator*() const { return this->iter.operator*(); }
cdi::pointer cdi::operator->() { return this->iter.operator->(); }

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

CheckedDFSView::CheckedDFSView(IDiGraphView const *g, std::unordered_set<Node> const &starting_points) 
  : graph(g), starting_points(starting_points)
{ }

checked_dfs_iterator CheckedDFSView::cbegin() const {
  return checked_dfs_iterator(*this->graph, this->starting_points);
}

checked_dfs_iterator CheckedDFSView::cend() const {
  return checked_dfs_iterator(*this->graph, {}, get_nodes(*this->graph));
}

checked_dfs_iterator CheckedDFSView::begin() const {
  return this->cbegin();
}

checked_dfs_iterator CheckedDFSView::end() const {
  return this->cend();
}

CheckedDFSView dfs(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  return CheckedDFSView(&g, starting_points);
}

UncheckedDFSView::UncheckedDFSView(IDiGraphView const *g, std::unordered_set<Node> const &starting_points) 
  : graph(g), starting_points(starting_points)
{ }

unchecked_dfs_iterator UncheckedDFSView::cbegin() const {
  return unchecked_dfs_iterator(*this->graph, this->starting_points);
}

unchecked_dfs_iterator UncheckedDFSView::cend() const {
  return unchecked_dfs_iterator(*this->graph, std::vector<Node>{});
}

unchecked_dfs_iterator UncheckedDFSView::begin() const {
  return this->cbegin();
}

unchecked_dfs_iterator UncheckedDFSView::end() const {
  return this->cend();
}

UncheckedDFSView unchecked_dfs(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  return UncheckedDFSView(&g, starting_points);
}

}
}
