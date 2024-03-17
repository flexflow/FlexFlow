#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_MACROS_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_MACROS_H

#include "utils/fmt_extra/is_fmtable.h"
#include "utils/type_traits_extra/is_streamable.h"
#include <ostream>
#include <string>

#define MAKE_TYPEDEF_PRINTABLE(TYPEDEF_NAME, TYPEDEF_SHORTNAME)                \
  namespace fmt {                                                              \
  template <>                                                                  \
  struct formatter<TYPEDEF_NAME> : formatter<::std::string> {                  \
    template <typename FormatContext>                                          \
    auto format(TYPEDEF_NAME const &x, FormatContext &ctx) const               \
        -> decltype(ctx.out()) {                                               \
      ::std::string s = fmt::format("{}({})", (TYPEDEF_SHORTNAME), x.value()); \
      return formatter<::std::string>::format(s, ctx);                         \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  static_assert(true, "")

#define FF_TYPEDEF_PRINTABLE(TYPEDEF_NAME, TYPEDEF_SHORTNAME)                  \
  }                                                                            \
  MAKE_TYPEDEF_PRINTABLE(::FlexFlow::TYPEDEF_NAME, TYPEDEF_SHORTNAME);         \
  namespace FlexFlow {                                                         \
  inline std::ostream &operator<<(std::ostream &s, TYPEDEF_NAME const &t) {    \
    return (s << fmt::to_string(t));                                           \
  }                                                                            \
  static_assert(is_fmtable_v<TYPEDEF_NAME>);                                   \
  static_assert(is_streamable<TYPEDEF_NAME>::value);

#endif
