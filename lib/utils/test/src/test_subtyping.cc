#include "test/utils/doctest.h"
#include "utils/subtyping.h"

using namespace FlexFlow;

struct Parent {};
struct Child {
  operator Parent() const {
    return Parent{};
  }
};
struct Grandchild {
  operator Child() const {
    return Child{};
  }
};

MAKE_SUBTYPING_SYSTEM(test_subtyping);
MAKE_ROOT_SUBTYPING_TAG(test_subtyping, Parent);
MAKE_SUBTYPING_TAG(test_subtyping, Child, Parent);
MAKE_SUBTYPING_TAG(test_subtyping, Grandchild, Child);

enum which_called_t { PARENT, CHILD, GRANDCHILD };

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_subtyping<Parent>) {
  return PARENT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_subtyping<Child>) {
  return CHILD;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_subtyping<Grandchild>) {
  return GRANDCHILD;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_subtyping<Parent>) {
  return PARENT;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_subtyping<Child>) {
  return CHILD;
}

template <typename T>
which_called_t test_coerce_up_two_levels(T const &t, test_subtyping<Parent>) {
  return PARENT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t) {
  return test_coerce_to_self(t, create_tag(t));
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t) {
  return test_coerce_up_one_level(t, create_tag(t));
}

template <typename T>
which_called_t test_coerce_up_two_levels(T const &t) {
  return test_coerce_up_two_levels(t, create_tag(t));
}

TEST_CASE("subtyping - coerce") {
  Grandchild g; 

  CHECK(test_coerce_to_self(g) == GRANDCHILD);
  CHECK(test_coerce_up_one_level(g) == CHILD);
  CHECK(test_coerce_up_two_levels(g) == PARENT);

  CHECK(test_coerce_to_self(coerce<test_subtyping<Child>>(g)) == CHILD);
  CHECK(test_coerce_to_self(coerce<test_subtyping<Parent>>(g)) == PARENT);
}
