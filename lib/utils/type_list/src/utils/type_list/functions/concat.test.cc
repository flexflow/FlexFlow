#include "utils/type_list/functions/concat.h"
#include "utils/testing.h"

struct t1 {};
struct t2 {};
struct t3 {};

template <typename... Ts>
using args = std::tuple<Ts...>;

TEST_SUITE(FF_TEST_SUITE) {
  // clang-format off
  TEST_CASE_TEMPLATE_T("type_list_concat_t",
                       WRAP_ARG(L_IN, R_IN, OUT),
                       args< type_list<>   , type_list<>           , type_list<>               >,
                       args< type_list<t1> , type_list<>           , type_list<t1>             >,
                       args< type_list<>   , type_list<t2>         , type_list<t2>             >,
                       args< type_list<t2> , type_list<t1>         , type_list<t2, t1>         >,
                       args< type_list<t2> , type_list<t2>         , type_list<t2, t2>         >,
                       args< type_list<t2> , type_list<t1, t2, t3> , type_list<t2, t1, t2, t3> >
                       ) {
    using result = type_list_concat_t<L_IN, R_IN>;
    using correct = OUT;
    CHECK_TYPE_EQ(result, correct);
  }
  // clang-format on
}
