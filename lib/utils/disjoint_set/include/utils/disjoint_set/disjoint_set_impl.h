#ifndef _FLEXFLOW_LIB_UTILS_DISJOINT_SET_INCLUDE_UTILS_DISJOINT_SET_DISJOINT_SET_IMPL_H
#define _FLEXFLOW_LIB_UTILS_DISJOINT_SET_INCLUDE_UTILS_DISJOINT_SET_DISJOINT_SET_IMPL_H

#include "utils/strong_typedef/strong_typedef.h"
#include "utils/type_traits_extra/is_rc_copy_compliant.h"
#include "utils/visitable.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

struct disjoint_set_node_t {
  size_t uid;
  size_t rank;
};
FF_VISITABLE_STRUCT(disjoint_set_node_t, uid, rank);

struct disjoint_set_impl {
  disjoint_set_impl(disjoint_set_impl const &) = delete;
  disjoint_set_impl &operator=(disjoint_set_impl const &) = delete;

  virtual void m_union(disjoint_set_node_t const &lhs,
                       disjoint_set_node_t const &rhs) = 0;
  virtual disjoint_set_node_t find(disjoint_set_node_t const &) const = 0;
  virtual disjoint_set_impl *clone() const = 0;
  virtual ~disjoint_set_impl() = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(disjoint_set_impl);

} // namespace FlexFlow

#endif
