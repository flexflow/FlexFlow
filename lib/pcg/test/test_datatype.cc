#include "doctest.h"
#include "pcg/file_format/v1/datatype.h"
#include "utils.h"

using namespace FlexFlow;

#define TEST_MEMBER(m, exp)                                                    \
  do {                                                                         \
    V1DataType v10 = to_v1(m);                                                 \
    CHECK(from_v1(v10) == m);                                                  \
    CHECK(str(json(v10)) == "\"" exp "\"");                                    \
  } while (0)

TEST_CASE("DataType") {
  TEST_MEMBER(DataType::BOOL, "BOOL");
  TEST_MEMBER(DataType::INT32, "INT32");
  TEST_MEMBER(DataType::INT64, "INT64");
  TEST_MEMBER(DataType::HALF, "HALF");
  TEST_MEMBER(DataType::FLOAT, "FLOAT");
  TEST_MEMBER(DataType::DOUBLE, "DOUBLE");
}

TEST_CASE("DataTypeValue") {
  DataTypeValue b = true;
  DataTypeValue i32 = (int32_t)32;
  DataTypeValue i64 = (int64_t)64L;
  DataTypeValue f16 = half(3.14);
  DataTypeValue f32 = (float)2.71828;
  DataTypeValue f64 = (double)1.414235;

  CHECK(from_v1(to_v1(b)) == b);
  CHECK(from_v1(to_v1(i32)) == i32);
  CHECK(from_v1(to_v1(i64)) == i64);
  CHECK(from_v1(to_v1(f16)) == f16);
  CHECK(from_v1(to_v1(f32)) == f32);
  CHECK(from_v1(to_v1(f64)) == f64);
}
