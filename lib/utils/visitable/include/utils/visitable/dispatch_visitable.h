#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_MAKE_VISITABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_MACROS_MAKE_VISITABLE_H

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/selection/min.hpp>
#include <boost/preprocessor/variadic/size.hpp>

#define _DISPATCH_MAYBE_EMPTY_CASE0(MACRO) BOOST_PP_CAT(MACRO, _EMPTY)
#define _DISPATCH_MAYBE_EMPTY_CASE1(MACRO) BOOST_PP_CAT(MACRO, _NONEMPTY)

#define _DISPATCH_FUNC_NAME(...)                                        \
      BOOST_PP_CAT(_DISPATCH_MAYBE_EMPTY_CASE,                                 \
                   BOOST_PP_MIN(BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), 1))

#define _DISPATCH_VISITABLE(MACRO, TYPENAME, ...)                              \
      _DISPATCH_FUNC_NAME(__VA_ARGS__)(MACRO)(TYPENAME, __VA_ARGS__)

#endif
