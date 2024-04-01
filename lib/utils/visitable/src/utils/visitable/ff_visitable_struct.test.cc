#include "utils/testing.h"
#include "utils/visitable/ff_visitable_struct.h"
#include "utils/visitable/is_visitable.h"

namespace FlexFlow {
struct example_t_nonempty {
  int field0;
  float field1;
};
FF_VISITABLE_STRUCT(example_t_nonempty, field0, field1);

struct example_t_empty { };
FF_VISITABLE_STRUCT(example_t_empty);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("FF_VISITABLE_STRUCT(example_t_nonempty, ...)") {
    CHECK(is_visitable_v<example_t_nonempty>);
  }

  TEST_CASE("FF_VISITABLE_STRUCT(example_t_empty)") {
    CHECK(is_visitable_v<example_t_empty>);
  }
}
