#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FMT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FMT_H

#include "utils/fmt_extra/is_fmtable.h"
#include "utils/type_traits_extra/metafunction/elements_satisfy.h"
#include "utils/visitable/is_visitable.h"
#include <sstream>

namespace FlexFlow {

struct fmt_visitor {
  std::ostringstream &oss;

  template <typename T>
  void operator()(char const *field_name, T const &field_value) {
    oss << " " << field_name << "=" << field_value;
  }
};

template <typename T>
std::string visit_format(T const &t) {
  static_assert(is_visitable<T>::value,
                "visit_format can only be applied to visitable types");
  static_assert(elements_satisfy<is_fmtable, T>::value,
                "Visitable fields must be fmtable");

  std::ostringstream oss;
  oss << "<" << ::visit_struct::get_name<T>();
  visit_struct::for_each(t, fmt_visitor{oss});
  oss << ">";

  return oss.str();
}

} // namespace FlexFlow

#endif
