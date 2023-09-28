#include "utils/testing.h"
#include "utils/preprocessor_extra/template.h"
#include "utils/preprocessor_extra/stringize.h"

TEST_CASE("TEMPLATE_DECL") {
  CHECK(STRINGIZE(TEMPLATE_DECL(4)) == "typename T0 , typename T1 , typename T2 , typename T3");
  CHECK(STRINGIZE(TEMPLATE_DECL(1)) == "typename T0");
  CHECK(STRINGIZE(TEMPLATE_DECL(0)) == "");
}

TEST_CASE("TEMPLATE_SPECIALIZE") {
  CHECK(STRINGIZE(TEMPLATE_SPECIALIZE(X, 4)) == "X< T0 , T1 , T2 , T3>");
  CHECK(STRINGIZE(TEMPLATE_SPECIALIZE(X, 1)) == "X< T0>");
  CHECK(STRINGIZE(TEMPLATE_SPECIALIZE(X, 0)) == "X");
}
