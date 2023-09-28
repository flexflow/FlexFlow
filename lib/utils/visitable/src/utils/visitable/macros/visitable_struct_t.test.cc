#include "utils/testing.h"
#include "utils/visitable/macros/visitable_struct_t.h"
#include "utils/visitable/type/functions/visit_as_tuple.h"
#include <tuple>
#include "utils/type_traits_extra/debug_print_type.h"
#include "utils/visitable/type/functions/type_at.h"
#include "utils/visitable/type/functions/get_name.h"
#include "utils/visitable/type/functions/field_count.h"
#include "utils/visitable/operations/eq.h"
#include "utils/visitable/type/functions/as_type_list.h"
#include "utils/type_traits_extra/is_equal_comparable.h"

namespace FlexFlow {
  struct example_t_0 {
    int my_first_field;
    float my_second_field;
  };
}
VISITABLE_STRUCT_T(example_t_0, 0, my_first_field, my_second_field);

TEST_CASE("VISITABLE_STRUCT_T<>") {
  CHECK(get_name<example_t_0>() == "example_t_0");
  CHECK_SAME_TYPE(type_at_t<0, example_t_0>, int);
  CHECK_SAME_TYPE(type_at_t<1, example_t_0>, float);
  CHECK(field_count_v<example_t_0> == 2);
  CHECK_SAME_TYPE(visit_as_tuple_t<example_t_0>, std::tuple<int, float>);
  CHECK(is_equal_comparable_v<example_t_0>);

  example_t_0 v1 { 1, 1.0f };
  example_t_0 v2 { 2, 2.0f };
  CHECK(v1 == v1);
  CHECK(v2 == v2);
  CHECK_FALSE(v1 == v2);
}

namespace FlexFlow {
  template <typename T>
  struct example_t_1 {
    T my_first_field;
    T my_second_field;
  };
}
VISITABLE_STRUCT_T(example_t_1, 1, my_first_field, my_second_field);

TEST_CASE_TEMPLATE("VISITABLE_STRUCT_T<T0>", T, int, std::string, std::vector<float>, example_t_1<std::string>) {
  CHECK(get_name<example_t_1<T>>() == "example_t_1");
  CHECK_SAME_TYPE(type_at_t<0, example_t_1<T>>, T);
  CHECK_SAME_TYPE(type_at_t<1, example_t_1<T>>, T);
  CHECK(field_count_v<example_t_1<T>> == 2);
  CHECK_SAME_TYPE(visit_as_tuple_t<example_t_1<T>>, std::tuple<T, T>);
}

namespace FlexFlow {
  template <typename T1, typename T2>
  struct example_t_2 {
    T1 my_first_field;
    T2 my_second_field;
  };
}
VISITABLE_STRUCT_T(example_t_2, 2, my_first_field, my_second_field);

TEST_CASE("VISITABLE_STRUCT_T<T0, T1>") {
  using T0 = int;
  using T1 = float;

  CHECK(get_name<example_t_2<T0, T1>>() == "example_t_2");
  CHECK_SAME_TYPE(type_at_t<0, example_t_2<T0, T1>>, T0);
  CHECK_SAME_TYPE(type_at_t<1, example_t_2<T0, T1>>, T1);
  CHECK(field_count_v<example_t_2<T0, T1>> == 2);
  CHECK_SAME_TYPE(visit_as_tuple_t<example_t_2<T0, T1>>, std::tuple<T0, T1>);
  CHECK(is_visitable_v<example_t_2<T0, T1>>);
  CHECK(elements_satisfy_v<is_equal_comparable, example_t_2<T0, T1>>);

  REQUIRE(is_equal_comparable_v<T0>);
  REQUIRE(is_equal_comparable_v<T1>);
  static_assert(is_equal_comparable<example_t_2<T0, T1>>::value);

  example_t_2<T0, T1> v1 { 1, 1.0f };
  example_t_2<T0, T1> v2 { 2, 2.0f };
  CHECK(v1 == v1);
  CHECK(v2 == v2);
  CHECK_FALSE(v1 == v2);
}

struct opaque_type_0 { };
struct opaque_type_1 { };
struct opaque_type_2 { };
struct opaque_type_3 { };
struct opaque_type_4 { };
struct opaque_type_5 { };
struct opaque_type_6 { };

template <typename T0, 
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
struct example_t_7 {
  T0 field_0;
  T1 field_1;
  T2 field_2;
  T3 field_3;
  T4 field_4;
  T5 field_5;
  T6 field_6;
};
VISITABLE_STRUCT_T(example_t_7, 7, field_0, field_1, field_2, field_3, field_4, field_5, field_6);

TEST_CASE("VISITABLE_STRUCT_T<T0, T1, T2, T3, T4, T5, T6>") {
  using T = example_t_7<opaque_type_0, 
                        opaque_type_1,
                        opaque_type_2,
                        opaque_type_3,
                        opaque_type_4,
                        opaque_type_5,
                        opaque_type_6>;

  CHECK(get_name<T>() == "example_t_7");
  CHECK_SAME_TYPE(type_at_t<0, T>, opaque_type_0);
  CHECK_SAME_TYPE(type_at_t<1, T>, opaque_type_1);
  CHECK_SAME_TYPE(type_at_t<2, T>, opaque_type_2);
  CHECK_SAME_TYPE(type_at_t<3, T>, opaque_type_3);
  CHECK_SAME_TYPE(type_at_t<4, T>, opaque_type_4);
  CHECK_SAME_TYPE(type_at_t<5, T>, opaque_type_5);
  CHECK_SAME_TYPE(type_at_t<6, T>, opaque_type_6);
  CHECK(field_count_v<T> == 7);
  CHECK_SAME_TYPE(
      visit_as_tuple_t<T>, 
      std::tuple<
        opaque_type_0,
        opaque_type_1,
        opaque_type_2,
        opaque_type_3,
        opaque_type_4,
        opaque_type_5,
        opaque_type_6
      >
  );
}

template <typename T0, typename T1, typename T2 = T0, typename T3 = T1>
struct example_t_2_4 {
  T0 field_0;
  T1 field_1;
  T2 field_2;
  T3 field_3;
};
VISITABLE_STRUCT_T(example_t_2_4, 4, field_0, field_1, field_2, field_3);

template <typename T0, typename T1, typename T2 = T0, typename T3 = T1>
struct example_t_2_4_prime {
  T0 field_0;
  T1 field_1;
  T2 field_2;
  T3 field_3;
};
VISITABLE_STRUCT_T(example_t_2_4_prime, 2, field_0, field_1, field_2, field_3);


TEST_CASE("VISITALE_STRUCT_T<T0, T1, T2=T0, T3=T1>") {
  using T = example_t_2_4<opaque_type_0, opaque_type_1>;

  CHECK(get_name<T>() == "example_t_2_4");
  CHECK_SAME_TYPE(type_at_t<0, T>, opaque_type_0);
  CHECK_SAME_TYPE(type_at_t<1, T>, opaque_type_1);
  CHECK_SAME_TYPE(type_at_t<2, T>, opaque_type_0);
  CHECK_SAME_TYPE(type_at_t<3, T>, opaque_type_1);
  CHECK(field_count_v<T> == 4);
  CHECK_SAME_TYPE(
      visit_as_tuple_t<T>, 
      std::tuple<
        opaque_type_0,
        opaque_type_1,
        opaque_type_0,
        opaque_type_1
      >
  );

  using T2 = example_t_2_4_prime<opaque_type_0, opaque_type_1>;
  CHECK(get_name<T2>() == "example_t_2_4_prime");
  CHECK_SAME_TYPE(type_at_t<0, T2>, opaque_type_0);
  CHECK_SAME_TYPE(type_at_t<1, T2>, opaque_type_1);
  CHECK_SAME_TYPE(type_at_t<2, T2>, opaque_type_0);
  CHECK_SAME_TYPE(type_at_t<3, T2>, opaque_type_1);
  CHECK(field_count_v<T2> == 4);
  CHECK_SAME_TYPE(
      visit_as_tuple_t<T2>, 
      std::tuple<
        opaque_type_0,
        opaque_type_1,
        opaque_type_0,
        opaque_type_1
      >
  );
}
