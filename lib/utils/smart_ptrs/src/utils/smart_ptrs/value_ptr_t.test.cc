#include "utils/smart_ptrs/value_ptr_t.h"
#include "utils/testing.h"

struct example_t {
  example_t *clone() {
    return new example_t{*this};
  }

  int x;
  float y;
};

TEST_CASE("value_ptr") {
  value_ptr<example_t> v =
      make_value_ptr<example_t>(3, static_cast<float>(5.0));
  CHECK(v->x == 3);
  CHECK(v->y == 5.0);
}

TEST_CASE("value_ptr copy semantics") {
  value_ptr<example_t> v =
      make_value_ptr<example_t>(3, static_cast<float>(5.0));
  auto v2 = v;
  v2->x = 4;

  CHECK(v->x == 3);
  CHECK(v2->x == 4);
}
