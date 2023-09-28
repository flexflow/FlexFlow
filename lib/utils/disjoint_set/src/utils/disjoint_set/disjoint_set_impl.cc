#include "utils/disjoint_set/disjoint_set_impl.h"

namespace FlexFlow {

disjoint_set_node_t &disjoint_set_impl::get_parent(disjoint_set_node_t const &n) const {
  if (this->parents.count(n) == 0) {
    this->parents.insert(n, n);
  } 

  return this->parents.at(n);
}

void disjoint_set_impl::m_union(disjoint_set_node_t const &lhs, disjoint_set_node_t const &rhs) {
  this->parents[this->find(lhs)] = this->find(rhs);
}

disjoint_set_node_t disjoint_set_impl::find(disjoint_set_node_t const &n) const {
  disjoint_set_node_t &parent = this->get_parent(n);

  if (parent == n) {
    return n;
  }

  parent = this->find(parent);
  return parent;
}

};
