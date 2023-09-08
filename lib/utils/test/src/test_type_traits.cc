#include "test/utils/all.h"
#include "utils/type_traits.h"

using namespace FlexFlow;

struct A {};
struct B {};

template <typename T> struct IsA : std::false_type {};
template <> struct IsA<A> : std::true_type {};

template <typename T> struct IsB : std::false_type {};
template <> struct IsB<B> : std::true_type {};

template <typename T> struct AlwaysFalse : std::false_type {};

template <typename T, typename Enable = void> struct FancyIsA : std::false_type {};

template <typename T>
struct FancyIsA<T, std::enable_if_t<std::is_same_v<T, A>>> : std::true_type {};

TEST_CASE("elements_satisfy") {
  CHECK(elements_satisfy<AlwaysFalse, std::tuple<>>::value);
  CHECK(elements_satisfy<IsA, std::tuple<A>>::value);    
  CHECK(elements_satisfy<FancyIsA, std::tuple<A>>::value);
  CHECK_FALSE(elements_satisfy<IsB, std::tuple<A>>::value);
  CHECK_FALSE(elements_satisfy<IsB, std::tuple<A, B>>::value);
  CHECK_FALSE(elements_satisfy<IsB, std::tuple<B, A>>::value);
  CHECK_FALSE(elements_satisfy<FancyIsA, std::tuple<B, A>>::value);
}

TEST_CASE("violating_element") {
  CHECK_SAME_TYPE(violating_element_t<AlwaysFalse, std::tuple<>>, void);
  CHECK_SAME_TYPE(violating_element_t<IsA, std::tuple<A>>, void);
  CHECK_SAME_TYPE(violating_element_t<FancyIsA, std::tuple<A>>, void);
  CHECK_SAME_TYPE(violating_element_t<IsB, std::tuple<A>>, A);
  CHECK_SAME_TYPE(violating_element_t<IsB, std::tuple<A, B>>, A);
  CHECK_SAME_TYPE(violating_element_t<IsB, std::tuple<B, A>>, A);
  CHECK_SAME_TYPE(violating_element_t<FancyIsA, std::tuple<B, A>>, B);
}
