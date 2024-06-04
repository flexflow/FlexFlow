#include "test/utils/doctest.h"
#include "op-attrs/datatype.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("can_promote_datatype_from_to(DataType, DataType)") {
    CHECK(can_strictly_promote_datatype_from_to(DataType::BOOL, DataType::INT32));
    CHECK(can_strictly_promote_datatype_from_to(DataType::INT32, DataType::INT64));
    CHECK(can_strictly_promote_datatype_from_to(DataType::FLOAT, DataType::DOUBLE));

    SUBCASE("is strict") {
      rc::check([](DataType d) {
        RC_ASSERT(!can_strictly_promote_datatype_from_to(d, d));
      });
    }

    SUBCASE("is asymmetric") {
      rc::check([](DataType l, DataType r) {
        RC_PRE(can_strictly_promote_datatype_from_to(l, r));
        RC_ASSERT(!can_strictly_promote_datatype_from_to(r, l));
      });
    }
  }
}
