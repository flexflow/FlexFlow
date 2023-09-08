#include "testing.h"
#include "utils/type_traits_extra/is_static_castable.h"

using namespace FlexFlow;

struct B {};

struct A {
  explicit operator B();
};

TEST_CASE_TEMPLATE("is_static_castable", 
                   std::pair<A, B>) {
  CHECK(is_static_castable<A, B>);
}
