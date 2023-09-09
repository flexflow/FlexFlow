#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_DEBUG_PRINT_TYPE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_DEBUG_PRINT_TYPE_H

#include <type_traits>
#include <boost/preprocessor/variadic/to_tuple.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>

namespace FlexFlow {

template <typename T>
struct ADDITIONAL_ERROR_INFO {};

struct ALL_OK {
  using expected_to_not_exist_to_allow_printing_the_following = int;
};

template <bool b, typename T>
struct print_on_condition;

template <bool DidSucceed, typename T>
using print_on_condition_t = std::conditional_t<DidSucceed, ALL_OK, T>;

}

#define WRAP_ARG(...) \
  BOOST_PP_VARIADIC_TO_TUPLE(__VA_ARGS__)

#define UNWRAP_ARG(ARGNAME) \
  BOOST_PP_REMOVE_PARENS(ARGNAME)

#define DEBUG_PRINT_TYPE(...) \
  using debug_print_proxy_type_2 = __VA_ARGS__; \
  using debug_print_proxy_type =                                                                \
      typename debug_print_proxy_type_2 ::expected_to_not_exist_to_allow_printing_the_following

#define ERROR_PRINT_TYPE(cond, msg, ...) \
  DEBUG_PRINT_TYPE(print_on_condition_t<UNWRAP_ARG(cond), ADDITIONAL_ERROR_INFO<__VA_ARGS__>>)

#endif
