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
  void m_union(optional<T> const &l, optional<T> const &r) {
    this->add_node_if_missing(l);
    this->add_node_if_missing(r);
    optional<T> const ll = this->find(l);
    optional<T> const rr = this->find(r);
    if (ll != rr) {
      this->mapping[ll] = rr;
    }
  }

  optional<T> const find(optional<T> const &t) const {
    this->add_node_if_missing(t);
    optional<T> const parent = this->mapping.at(t);
    if (!parent.has_value()) {
      return t;
    } else {
      return this->find(parent);
    }
  }

private:
  void add_node_if_missing(optional<T> const &t) const {
    if (mapping.find(t) == mapping.end()) {
      mapping[t] = nullopt;
    }
  }
  mutable std::unordered_map<optional<T>, optional<T>> mapping;
};

// Custom comparator for optional
template <typename T>
struct OptionalComparator {
  bool operator()(optional<T> const &lhs, optional<T> const &rhs) const {
    if (!lhs.has_value() || !rhs.has_value()) {
      return false;
    }
    return *lhs < *rhs;
  }
};

template <typename T, typename Compare = OptionalComparator<T>>
class disjoint_set {
public:
  void m_union(optional<T> const &l, optional<T> const &r) const {
    this->nodes.insert(l);
    this->nodes.insert(r);
    this->ds.m_union(this->get_node(l), this->get_node(r));
  }

  optional<T> const find(optional<T> const &t) const {
    this->nodes.insert(t); // Make sure the node is in the set
    return this->ds.find(this->get_node(t));
  }

  std::map<optional<T>, optional<T>, Compare> get_mapping() const {
    std::map<optional<T>, optional<T>, Compare> mapping;
    for (optional<T> const &t : this->nodes) {
      mapping[t] = this->ds.find(t);
    }
    return mapping;
  }

private:
  optional<T> const get_node(optional<T> const &t) const {
    auto it = this->nodes.find(t);
    assert(it != this->nodes.end());
    return *it;
  }

  mutable m_disjoint_set<T> ds;
  mutable std::set<optional<T>, Compare>
      nodes; // Note(lambda): make mutable to allow using ds->find() to be const
             // because while the result is invariant to path compression, etc.
};

} // namespace FlexFlow

#endif
