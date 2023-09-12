#ifndef _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_TEMPLATE_ARGS_H
#define _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_TEMPLATE_ARGS_H

#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include "utils/preprocessor_extra/wrap_arg.h"

#define TEMPLATE_DECL(N) \
  BOOST_PP_ENUM_PARAMS(N, typename T)

#define TEMPLATE_SPECIALIZE(TYPENAME, N) \
   UNWRAP_ARG(BOOST_PP_IIF(BOOST_PP_GREATER(N, 0), WRAP_ARG(TYPENAME<BOOST_PP_ENUM_PARAMS(N, T)>), WRAP_ARG(TYPENAME)))

#endif
