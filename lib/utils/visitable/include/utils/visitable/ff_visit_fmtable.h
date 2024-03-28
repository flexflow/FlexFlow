#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_FF_VISIT_FMTABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_FF_VISIT_FMTABLE_H

#include "utils/visitable/visit_format.h"

namespace FlexFlow {

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
