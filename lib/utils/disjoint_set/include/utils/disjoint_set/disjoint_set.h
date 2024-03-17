#ifndef _FLEXFLOW_DISJOINT_SET_H
#define _FLEXFLOW_DISJOINT_SET_H

#include "disjoint_set_impl.h"
#include "utils/bidict/bidict.h"
#include "utils/smart_ptrs/cow_ptr_t.h"
#include <unordered_map>

namespace FlexFlow {

template <typename T>
struct disjoint_set {
public:
  void m_union(T const &l, T const &r) {
    this->impl.get_mutable()->m_union(this->at(l), this->at(r));
  }

  T const &find(T const &t) const {
    disjoint_set_node_t t_node = this->at(t);
    disjoint_set_node_t found_node = this->impl->find(t_node);
    return this->node_mapping.at_r(found_node);
  }

private:
  disjoint_set_node_t at(T const &t) const {
    if (!contains(node_mapping, t)) {
      node_mapping.equate(t, this->fresh());
    }
    return node_mapping.at_l(t);
  }

  disjoint_set_node_t fresh() const {
    return {++this->node_ctr};
  }

  cow_ptr_t<disjoint_set_impl> impl;
  mutable bidict<T, disjoint_set_node_t> node_mapping;
  mutable size_t node_ctr;
};

} // namespace FlexFlow

#endif
