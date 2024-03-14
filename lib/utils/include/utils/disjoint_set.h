#ifndef _FLEXFLOW_DISJOINT_SET_H
#define _FLEXFLOW_DISJOINT_SET_H

#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace FlexFlow {

template <typename T>
class m_disjoint_set {
public:
  void m_union(std::optional<T> const &l, std::optional<T> const &r) {
    this->add_node_if_missing(l);
    this->add_node_if_missing(r);
    std::optional<T> const ll = this->find(l);
    std::optional<T> const rr = this->find(r);
    if (ll != rr) {
      this->mapping[ll] = rr;
    }
  }

  std::optional<T> const find(std::optional<T> const &t) const {
    this->add_node_if_missing(t);
    std::optional<T> const parent = this->mapping.at(t);
    if (!parent.has_value()) {
      return t;
    } else {
      return this->find(parent);
    }
  }

private:
  void add_node_if_missing(std::optional<T> const &t) const {
    if (mapping.find(t) == mapping.end()) {
      mapping[t] = std::nullopt;
    }
  }
  mutable std::unordered_map<std::optional<T>, std::optional<T>> mapping;
};

// Custom comparator for optional
template <typename T>
struct OptionalComparator {
  bool operator()(std::optional<T> const &lhs, std::optional<T> const &rhs) const {
    if (!lhs.has_value() || !rhs.has_value()) {
      return false;
    }
    return *lhs < *rhs;
  }
};

template <typename T, typename Compare = OptionalComparator<T>>
class disjoint_set {
public:
  void m_union(std::optional<T> const &l, std::optional<T> const &r) const {
    this->nodes.insert(l);
    this->nodes.insert(r);
    this->ds.m_union(this->get_node(l), this->get_node(r));
  }

  std::optional<T> const find(std::optional<T> const &t) const {
    this->nodes.insert(t); // Make sure the node is in the set
    return this->ds.find(this->get_node(t));
  }

  std::map<std::optional<T>, std::optional<T>, Compare> get_mapping() const {
    std::map<std::optional<T>, std::optional<T>, Compare> mapping;
    for (std::optional<T> const &t : this->nodes) {
      mapping[t] = this->ds.find(t);
    }
    return mapping;
  }

private:
  std::optional<T> const get_node(std::optional<T> const &t) const {
    auto it = this->nodes.find(t);
    assert(it != this->nodes.end());
    return *it;
  }

  mutable m_disjoint_set<T> ds;
  mutable std::set<std::optional<T>, Compare>
      nodes; // Note(lambda): make mutable to allow using ds->find() to be const
             // because while the result is invariant to path compression, etc.
};

} // namespace FlexFlow

#endif
