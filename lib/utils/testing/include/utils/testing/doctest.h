#include "doctest/doctest.h"
#include <type_traits>

namespace doctest {

#define CHECK_SAME_TYPE(...) CHECK(std::is_same_v<__VA_ARGS__>);

} // namespace doctest
