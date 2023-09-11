#include "utils/testing.h"
#include "utils/smart_ptrs/value_ptr_t.h"

struct example_t {
  example_t *clone() {
    return new example_t{*this};
  }

  int x;
  float y;
};

TEST_CASE("value_ptr") {
  value_ptr<example_t> v = make_value_ptr<example_t>(3, 5.0); 
  CHECK(v->x == 3);
  CHECK(v->y == 5.0);
}
