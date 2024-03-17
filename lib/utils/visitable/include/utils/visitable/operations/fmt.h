#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FMT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FMT_H

#include "utils/fmt_extra/is_fmtable.h"
#include "utils/type_traits_extra/metafunction/elements_satisfy.h"
#include "utils/visitable/type/traits/is_visitable.h"
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

template <typename T>
struct visitable_formatter : public ::fmt::formatter<std::string> {
  template <typename FormatContext>
  auto format(T const &t, FormatContext &ctx) const -> decltype(ctx.out()) {
    std::string fmted = visit_format(t);
    return formatter<std::string>::format(fmted, ctx);
  }
};

#define FF_VISIT_FMTABLE(TYPENAME)                                             \
  static_assert(is_visitable<TYPENAME>::value,                                 \
                #TYPENAME " must be visitable to use FF_VISIT_FMTABLE");       \
  static_assert(THE_FAILING_ELEMENT_IS<                                        \
                    violating_element_t<is_streamable, TYPENAME>>::value,      \
                #TYPENAME "'s elements must be streamable");                   \
  }                                                                            \
  namespace fmt {                                                              \
  template <>                                                                  \
  struct formatter<::FlexFlow::TYPENAME>                                       \
      : ::FlexFlow::visitable_formatter<::FlexFlow::TYPENAME> {};              \
  }                                                                            \
  namespace FlexFlow {                                                         \
  static_assert(is_fmtable_v<TYPENAME>,                                        \
                #TYPENAME                                                      \
                " failed sanity check on is_fmtable in FF_VISIT_FMTABLE");     \
  static_assert(is_streamable<TYPENAME>::value,                                \
                #TYPENAME                                                      \
                " failed sanity check on is_streamable in FF_VISIT_FMTABLE");

} // namespace FlexFlow

#endif
