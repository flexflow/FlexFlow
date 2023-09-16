#ifndef _FLEXFLOW_LIB_UTILS_FF_EXCEPTIONS_INCLUDE_UTILS_FF_EXCEPTIONS_TYPE_NOT_IMPLEMENTED_H
#define _FLEXFLOW_LIB_UTILS_FF_EXCEPTIONS_INCLUDE_UTILS_FF_EXCEPTIONS_TYPE_NOT_IMPLEMENTED_H

namespace FlexFlow {

struct type_function_not_implemented { 
#ifdef FF_REQUIRE_IMPLEMENTEED
  static_assert(false,  "Type not yet implemented");
#else
#endif
};

} // namespace FlexFlow

#endif
