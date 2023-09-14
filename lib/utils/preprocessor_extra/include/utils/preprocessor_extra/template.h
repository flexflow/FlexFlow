#ifndef _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_TEMPLATE_ARGS_H
#define _FLEXFLOW_LIB_UTILS_PREPROCESSOR_EXTRA_INCLUDE_UTILS_PREPROCESSOR_EXTRA_TEMPLATE_ARGS_H

#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_tuple.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/variadic/size.hpp>
#include <boost/preprocessor/tuple/eat.hpp>
#include <boost/preprocessor/tuple/push_front.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include "utils/preprocessor_extra/wrap_arg.h"

#define GET_TUPLE_ELEM(N, PREFIX, TUP) \
  PREFIX BOOST_PP_TUPLE_ELEM(N, TUP)

#define GET_TUPLE_ELEM_WRAPPER(z, CURR_REPITITION_NUM, PREFIX_AND_TUP) \
  BOOST_PP_EXPAND(GET_TUPLE_ELEM BOOST_PP_TUPLE_PUSH_FRONT(PREFIX_AND_TUP, CURR_REPITITION_NUM))
                              //
#define TEMPLATE_DECL_TUPLE(TUP) \
  BOOST_PP_ENUM(BOOST_PP_TUPLE_SIZE(TUP), GET_TUPLE_ELEM_WRAPPER, WRAP_ARG(typename, TUP))

#define TEMPLATE_SPECIALIZE_TUPLE(TYPENAME, TUP) \
    TYPENAME<UNWRAP_ARG(TUP)> 

#define TEMPLATE_DECL_N(N) \
  BOOST_PP_ENUM_PARAMS(N, typename T)

#define TEMPLATE_DECL_V(...) \
  TEMPLATE_DECL_TUPLE(WRAP_ARG(__VA_ARGS__))

#define TEMPLATE_SPECIALIZE_N(TYPENAME, N) \
   UNWRAP_ARG(BOOST_PP_IIF(BOOST_PP_GREATER(N, 0), WRAP_ARG(TYPENAME<BOOST_PP_ENUM_PARAMS(N, T)>), WRAP_ARG(TYPENAME)))

#define TEMPLATE_SPECIALIZE_V(TYPENAME, ...) \
  TEMPLATE_SPECIALIZE_TUPLE(TYPENAME, WRAP_ARG(__VA_ARGS__))

#endif
