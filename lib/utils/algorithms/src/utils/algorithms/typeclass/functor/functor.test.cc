#include "utils/testing.h"
#include "utils/algorithms/typeclass/functor/functor.h"

template <typename T>
struct opaque_container_type_t;

struct opaque_input_type_t;
struct opaque_output_type_t;

template <typename T>
struct opaque_functor {
  using A = T;

  template <typename X>
  using F = opaque_container_type_t<X>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(std::vector<A> const &v, Func const &f) {
  }

  template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<A, Func, A>>>
  static void fmap_inplace(std::vector<A> &v, Func const &f) {
  }
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("element_type_t") {
    CHECK_SAME_TYPE(element_type_t<opaque_functor<int>>, int);
  }
}
