#include "op-attrs/datatype.h"
#include "test/utils/doctest.h"
#include "test/utils/rapidcheck.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("can_promote_datatype_from_to(DataType, DataType)") {
    CHECK(
        can_strictly_promote_datatype_from_to(DataType::BOOL, DataType::INT32));
    CHECK(can_strictly_promote_datatype_from_to(DataType::INT32,
                                                DataType::INT64));
    CHECK(can_strictly_promote_datatype_from_to(DataType::FLOAT,
                                                DataType::DOUBLE));

    RC_SUBCASE("is strict", [](DataType d) {
      RC_ASSERT(!can_strictly_promote_datatype_from_to(d, d));
    });

    RC_SUBCASE("is asymmetric", [](DataType l, DataType r) {
      RC_PRE(can_strictly_promote_datatype_from_to(l, r));
      RC_ASSERT(!can_strictly_promote_datatype_from_to(r, l));
    });

    RC_SUBCASE("is transitive", [](DataType d1, DataType d2, DataType d3) {
      RC_PRE(can_strictly_promote_datatype_from_to(d1, d2));
      RC_PRE(can_strictly_promote_datatype_from_to(d2, d3));
      RC_ASSERT(can_strictly_promote_datatype_from_to(d1, d3));
    });
  }
}
