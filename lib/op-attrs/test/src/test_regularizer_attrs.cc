#include "doctest/doctest.h"
#include "op-attrs/regularizer_attrs.dtg.h"
#include <rapidcheck.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("RC") {
    CHECK(rc::check("valid variant", [](RegularizerAttrs reg) {
      return reg.has<L1RegularizerAttrs>() || reg.has<L2RegularizerAttrs>();
    }));
  }
}
