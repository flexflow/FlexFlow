#include "doctest/doctest.h"
#include "local-execution/ops/attention.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  
  TEST_CASE("fwd") {
    TaskImplFunction imf = get_attention_init_task_impl();
    InitTaskImplFunction fn = imf.get<InitTaskImplFunction>();
    std::string str = format_as(fn);

    CHECK(str == "");

    TaskImplFunction imf2 = get_attention_fwd_task_impl();
    FwdBwdTaskImplFunction fn2 = imf2.get<FwdBwdTaskImplFunction>();
    std::string str2 = format_as(fn2);

    CHECK(str2 == "");

    TaskImplFunction imf3 = get_attention_bwd_task_impl();
    FwdBwdTaskImplFunction fn3 = imf3.get<FwdBwdTaskImplFunction>();
    std::string str3 = format_as(fn3);

    CHECK(str3 == "");

    CHECK(fn2 == fn3);
  }


}

}
