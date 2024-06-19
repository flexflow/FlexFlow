#ifndef _FLEXFLOW_UTILS_GRAPH_TRAVERSAL_H
#define _FLEXFLOW_UTILS_GRAPH_TRAVERSAL_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/node.dtg.h"
#include <iterator>
#include <queue>
#include <vector>

namespace FlexFlow {

struct unchecked_dfs_iterator {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::size_t;
  using value_type = Node;
  using pointer = Node const *;
  using reference = Node const &;

  unchecked_dfs_iterator(DiGraphView const &g, std::vector<Node> const &);
  unchecked_dfs_iterator(DiGraphView const &g,
                         std::unordered_set<Node> const &);

  reference operator*() const;
  pointer operator->();

  // Prefix increment
  unchecked_dfs_iterator &operator++();

  // Postfix increment
  unchecked_dfs_iterator operator++(int);

  bool operator==(unchecked_dfs_iterator const &other) const;
  bool operator!=(unchecked_dfs_iterator const &other) const;

  void skip();

private:
  std::vector<Node> stack;
  DiGraphView graph;

  friend struct checked_dfs_iterator;
};

struct checked_dfs_iterator {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::size_t;
  using value_type = Node;
  using pointer = Node const *;
  using reference = Node const &;

  checked_dfs_iterator(DiGraphView const &g,
                       std::vector<Node> const &,
                       std::unordered_set<Node> const &);
  checked_dfs_iterator(DiGraphView const &g,
                       std::unordered_set<Node> const &starting_points);

  reference operator*() const;
  pointer operator->();
  checked_dfs_iterator &operator++();   // prefix increment
  checked_dfs_iterator operator++(int); // postfix increment

  bool operator==(checked_dfs_iterator const &) const;
  bool operator!=(checked_dfs_iterator const &) const;

private:
  unchecked_dfs_iterator iter;
  std::unordered_set<Node> seen;
};

struct bfs_iterator {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::size_t;
  using value_type = Node;
  using pointer = Node const *;
  using reference = Node const &;

  bfs_iterator(DiGraphView const &,
               std::queue<Node> const &,
               std::optional<std::unordered_set<Node>> const &);
  bfs_iterator(DiGraphView const &,
               std::unordered_set<Node> const &starting_points);

  reference operator*() const;
  pointer operator->();
  bfs_iterator &operator++();
  bfs_iterator operator++(int);

  bool operator==(bfs_iterator const &) const;
  bool operator!=(bfs_iterator const &) const;

private:
  DiGraphView graph;
  std::queue<Node> q;
  std::optional<std::unordered_set<Node>> seen;
};

struct CheckedDFSView {
  CheckedDFSView() = delete;
  explicit CheckedDFSView(DiGraphView const &,
                          std::unordered_set<Node> const &starting_points);

  checked_dfs_iterator begin() const;
  checked_dfs_iterator end() const;
  checked_dfs_iterator cbegin() const;
  checked_dfs_iterator cend() const;

private:
  DiGraphView graph;
  std::unordered_set<Node> starting_points;
};

struct UncheckedDFSView {
  UncheckedDFSView() = delete;
  explicit UncheckedDFSView(DiGraphView const &,
                            std::unordered_set<Node> const &starting_points);

  unchecked_dfs_iterator begin() const;
  unchecked_dfs_iterator end() const;
  unchecked_dfs_iterator cbegin() const;
  unchecked_dfs_iterator cend() const;

private:
  DiGraphView graph;
  std::unordered_set<Node> starting_points;
};

struct BFSView {
  BFSView() = delete;
  explicit BFSView(DiGraphView const &,
                   std::unordered_set<Node> const &starting_points);

  bfs_iterator begin() const;
  bfs_iterator end() const;
  bfs_iterator cbegin() const;
  bfs_iterator cend() const;

private:
  DiGraphView graph;
  std::unordered_set<Node> starting_points;
};

/* struct BoundaryDFSView { */
/*   BoundaryDFSView() = delete; */
/*   explicit BoundaryDFSView(IDiGraphView const *); */

/*   struct boundary_dfs_iterator { */
/*     using iterator_category = std::forward_iterator_tag; */
/*     using difference_type = std::size_t; */
/*     using value_type = Node; */
/*     using pointer = Node const *; */
/*     using reference = Node const &; */

/*     boundary_dfs_iterator(IDiGraphView const &g, std::vector<Node> const &,
 * std::unordered_set<Node> const &); */

/*     reference operator*() const; */
/*     pointer operator->(); */

/*     bool operator==(boundary_dfs_iterator const &other) const; */
/*     bool operator!=(boundary_dfs_iterator const &other) const; */
/*   private: */
/*     std::vector<Node> stack; */
/*     IDiGraphView const *graph; */
/*   }; */

/*   boundary_dfs_iterator begin() const; */
/*   boundary_dfs_iterator end() const; */
/*   boundary_dfs_iterator cbegin() const; */
/*   boundary_dfs_iterator cend() const; */
/* private: */
/*   IDiGraphView const *graph; */
/* }; */

UncheckedDFSView unchecked_dfs(DiGraphView const &,
                               std::unordered_set<Node> const &starting_points);
/* BoundaryDFSView boundary_dfs(IDiGraphView const &, std::unordered_set<Node>
 * const &starting_points); */
CheckedDFSView dfs(DiGraphView const &,
                   std::unordered_set<Node> const &starting_points);
BFSView bfs(DiGraphView const &,
            std::unordered_set<Node> const &starting_points);

} // namespace FlexFlow

#endif
