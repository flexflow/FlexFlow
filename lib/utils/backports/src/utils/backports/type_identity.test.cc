#include "utils/testing.h"
#include "utils/backports/type_identity.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("type_identity<T>", T, int, char) {
    CHECK_TYPE_EQ(typename type_identity<T>::type, type_identity_t<T>);
    CHECK_TYPE_EQ(type_identity_t<T>, T);
  }
}
