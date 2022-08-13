#ifndef _FLEXFLOW_DISJOINT_SET_H
#define _FLEXFLOW_DISJOINT_SET_H

#include <cassert>
#include <map>
#include <set>
#include <unordered_map>

template <typename T>
class m_disjoint_set {
public:
  void m_union(T const *l, T const *r) {
    this->add_node_if_missing(l);
    this->add_node_if_missing(r);
    T const *ll = this->find(l);
    T const *rr = this->find(r);
    if (ll != rr) {
      this->mapping[ll] = rr;
    }
  }
  T const *find(T const *t) {
    this->add_node_if_missing(t);
    T const *parent = this->mapping.at(t);
    if (parent == nullptr) {
      return t;
    } else {
      return this->find(parent);
    }
  }

private:
  void add_node_if_missing(T const *t) {
    if (mapping.find(t) == mapping.end()) {
      mapping[t] = nullptr;
    }
  }
  std::unordered_map<T const *, T const *> mapping;
};

template <typename T, typename Compare = std::less<T>>
class disjoint_set {
public:
  void m_union(T const &l, T const &r) {
    this->nodes.insert(l);
    this->nodes.insert(r);
    this->ds.m_union(this->get_node(l), this->get_node(r));
  }
  T const &find(T const &t) {
    this->nodes.insert(t);
    return *this->ds.find(this->get_node(t));
  }
  std::map<T, T, Compare> get_mapping() const {
    std::map<T, T, Compare> mapping;
    for (T const &t : this->nodes) {
      mapping[t] = this->ds.find(&t);
    }
    return mapping;
  }

private:
  T const *get_node(T const &t) {
    auto it = this->nodes.find(t);
    assert(it != this->nodes.end());
    return &*it;
  }

  m_disjoint_set<T> ds;
  std::set<T, Compare> nodes;
};

#endif // _FLEXFLOW_DISJOINT_SET_H
