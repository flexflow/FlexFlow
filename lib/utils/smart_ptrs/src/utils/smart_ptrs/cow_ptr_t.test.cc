#include "utils/testing.h"
#include "utils/smart_ptrs/cow_ptr_t.h"

struct clonable_t {
  clonable_t *clone() const { return new clonable_t{}; }
};

TEST_CASE("cow_ptr_t") {
  CHECK(make_cow_ptr<clonable_t>().get() != nullptr);
}
