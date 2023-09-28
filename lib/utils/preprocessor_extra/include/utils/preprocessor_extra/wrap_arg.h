#ifndef _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_WRAP_ARG_H
#define _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_WRAP_ARG_H

#include <boost/preprocessor/variadic/to_tuple.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>

#define WRAP_ARG(...) \
  BOOST_PP_VARIADIC_TO_TUPLE(__VA_ARGS__)

#define UNWRAP_ARG(ARGNAME) \
  BOOST_PP_REMOVE_PARENS(ARGNAME)

#endif
