#include "utils/testing.h"
#include "utils/fmt_extra/is_fmtable.h"

using namespace FlexFlow;

struct not_fmtable {};

TEST_CASE_TEMPLATE("is_fmtable", T, int) {
  CHECK(is_fmtable<T>::value == is_fmtable_v<T>);
  CHECK(is_fmtable_v<T>);
}

TEST_CASE_TEMPLATE("not is_fmtable", T, not_fmtable) {
  CHECK(is_fmtable<T>::value == is_fmtable_v<T>);
  CHECK_FALSE(is_fmtable_v<T>);
}
