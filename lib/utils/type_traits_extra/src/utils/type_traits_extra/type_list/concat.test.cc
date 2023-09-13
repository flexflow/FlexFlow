#include "utils/testing.h"
#include "utils/type_traits_extra/type_list/concat.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE_T("type_list_concat_t",
                       WRAP_ARG(L_IN, R_IN, OUT),
                       WRAP_ARG(type_list<>, type_list<>, type_list<>)) {
    CHECK(true);
  }
}
