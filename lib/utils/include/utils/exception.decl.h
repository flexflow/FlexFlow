#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_DECL_H

#include "utils/fmt.decl.h"
#include <stdexcept>

namespace FlexFlow {

#ifdef FF_REQUIRE_IMPLEMENTED
#define NOT_IMPLEMENTED() static_assert(false, "Function not yet implemented");
#else
#define NOT_IMPLEMENTED() throw not_implemented(__FILE__, __LINE__);
#endif

class not_implemented : public std::logic_error {
public:
  not_implemented(char const *file, unsigned line);
};

// This macro should only be used when code is known to be unreachable under
// normal circumstances but may get hit because of a *developer* (not user)
// error. An example of this would be adding a member to an enum but not
// updating all switch statements that use the enum. It is primarily provided
// to squelch compiler warnings about such situations.
#define NOT_REACHABLE() throw not_reachable(__FILE__, __LINE__);

class not_reachable : public std::logic_error {
public:
  not_reachable(char const *file, unsigned line);
};

template <typename... T>
std::runtime_error mk_runtime_error(fmt::format_string<T...> fmt_str,
                                    T &&...args);
} // namespace FlexFlow

#endif
