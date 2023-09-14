#ifndef _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_TEST_CASE_TEMPLATE_N_H
#define _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_TEST_CASE_TEMPLATE_N_H

#include "doctest/doctest.h"
#include "utils/preprocessor_extra/template.h"
#include "utils/preprocessor_extra/wrap_arg.h" 

namespace FlexFlow {

#define DOCTEST_TEST_CASE_TEMPLATE_T_DEFINE_IMPL(dec, T_TUP, iter, func)                                 \
    template <TEMPLATE_DECL_TUPLE(T_TUP)> \
    static void func();                                                                            \
    namespace { /* NOLINT */                                                                       \
        template <typename Tuple>                                                                  \
        struct iter;                                                                               \
        template <typename... ArgTypes, typename... Rest>                                                 \
        struct iter<std::tuple<std::tuple<ArgTypes...>, Rest...>>                                                     \
        {                                                                                          \
            iter(const char* file, unsigned line, int index) {                                     \
                std::ostringstream oss; \
                std::vector<doctest::String> arg_type_names = {doctest::toString<ArgTypes>()...}; \
                for (int i = 0; i < arg_type_names.size(); i++) { \
                  if (i != 0) { \
                    oss << ", "; \
                  } \
                  oss << arg_type_names.at(i); \
                } \
                std::string s = oss.str(); \
                doctest::String ss = doctest::toString(s); \
                doctest::detail::regTest(doctest::detail::TestCase(func<ArgTypes...>, file, line,         \
                                            doctest_detail_test_suite_ns::getCurrentTestSuite(),   \
                                            ss,                             \
                                            int(line) * 1000 + index)                              \
                                         * dec);                                                   \
                iter<std::tuple<Rest...>>(file, line, index + 1);                                  \
            }                                                                                      \
        };                                                                                         \
        template <>                                                                                \
        struct iter<std::tuple<>>                                                                  \
        {                                                                                          \
            iter(const char*, unsigned, int) {}                                                    \
        };                                                                                         \
    }                                                                                              \
    template <TEMPLATE_DECL_TUPLE(T_TUP)>                                                                          \
    static void func()


#define DOCTEST_TEST_CASE_TEMPLATE_T_IMPL(dec, T_TUP, anon, ...)                                         \
    DOCTEST_TEST_CASE_TEMPLATE_T_DEFINE_IMPL(dec, T_TUP, DOCTEST_CAT(anon, ITERATOR), anon);             \
    DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(anon, anon, std::tuple<__VA_ARGS__>)               \
    template <TEMPLATE_DECL_TUPLE(T_TUP)>                                                                          \
    static void anon()

#define DOCTEST_TEST_CASE_TEMPLATE_T(dec, T_TUP, ...)                                                    \
    DOCTEST_TEST_CASE_TEMPLATE_T_IMPL(dec, T_TUP, DOCTEST_ANONYMOUS(DOCTEST_ANON_TMP_), __VA_ARGS__)

#define TEST_CASE_TEMPLATE_T(name, T_TUP, ...) DOCTEST_TEST_CASE_TEMPLATE_T(name, T_TUP, __VA_ARGS__)

} // namespace FlexFlow

#endif
