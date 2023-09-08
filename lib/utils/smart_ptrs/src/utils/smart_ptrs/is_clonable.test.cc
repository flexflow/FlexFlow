#include "utils/testing.h"
#include "include/utils/smart_ptrs/is_clonable.h"

struct should_be_clonable {
  should_be_clonable *clone() const;
};

struct should_not_be_clonable { };

TEST_CASE("is_clonable") {
  CHECK(is_clonable<should_be_clonable>::value);
  CHECK_FALSE(is_clonable<should_not_be_clonable>::value);
}

TEST_CASE_TEMPLATE("is_clonable_v", T, should_be_clonable, should_not_be_clonable) {
  CHECK_IS_SAME(is_clonable<T>::value, is_clonable_v<T>);
}
