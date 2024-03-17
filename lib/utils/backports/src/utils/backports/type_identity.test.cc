#include "utils/backports/type_identity.h"
#include "testing.h"

TEMPLATE_TEST_CASE("type_identity", T, int, char) {
  CHECK_IS_SAME(typename type_identity<T>::type, type_identity_t<T>);
  CHECK_IS_SAME(type_identity_t<T>, T);
}
