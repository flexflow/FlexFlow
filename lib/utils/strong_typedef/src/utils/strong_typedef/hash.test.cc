#include "utils/testing.h"
#include "utils/strong_typedef/hash.h"
#include "utils/strong_typedef/strong_typedef.h"
#include "utils/type_traits_extra/is_hashable.h" 

template <typename T>
struct example_typedef_t : public strong_typedef<example_typedef_t<T>, T> {
  using strong_typedef<example_typedef_t<T>, T>::strong_typedef;
};

struct unhashable_t {};

TEST_CASE("hash<strong_typedef<T>>") {
  CHECK(is_hashable_v<example_typedef_t<int>>);
  CHECK_FALSE(is_hashable_v<example_typedef_t<unhashable_t>>);
}
