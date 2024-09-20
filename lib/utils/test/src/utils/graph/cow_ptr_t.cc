#include "utils/graph/cow_ptr_t.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace FlexFlow;

struct TestObject {
  TestObject(int x) : x(x) {}
  int x;
  virtual TestObject *clone() const {
    return new TestObject(x);
  }
};

struct TestObjectDerived : public TestObject {
  TestObjectDerived(int x, int y) : TestObject(x), y(y) {}
  int y;
  TestObjectDerived *clone() const override {
    return new TestObjectDerived(x, y);
  }
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cow_ptr_t constructor") {
    std::shared_ptr<TestObject> sp = std::make_shared<TestObject>(1);
    cow_ptr_t<TestObject> p1(sp);
    cow_ptr_t<TestObject> p2(std::make_shared<TestObject>(3));
    cow_ptr_t<TestObject> p3(TestObject(2));
    cow_ptr_t<TestObject> p4(p3);
    cow_ptr_t<TestObject> p5 = p1;
    CHECK(p1->x == 1);
    CHECK(p2->x == 3);
    CHECK(p3->x == 2);
    CHECK(p4->x == p3->x);
    CHECK(p5->x == p1->x);
  }

  TEST_CASE("cow_ptr_t copy") {
    cow_ptr_t<TestObject> p1(std::make_shared<TestObject>(1));
    cow_ptr_t<TestObject> p2(std::make_shared<TestObject>(2));
    p1 = p2;
    CHECK(p1->x == p2->x);
  }

  TEST_CASE("cow_ptr_t cast") {
    cow_ptr_t<TestObjectDerived> p1(std::make_shared<TestObjectDerived>(1, 2));
    cow_ptr_t<TestObject> p2(p1);
    CHECK(p2->x == 1);
  }

  TEST_CASE("cow_ptr_t get_mutable") {
    cow_ptr_t<TestObject> p1(std::make_shared<TestObject>(1));
    cow_ptr_t<TestObject> p2(p1);
    p1.get_mutable()->x = 3;
    CHECK(p1->x == 3);
    CHECK(p2->x == 1);
    p2.get_mutable()->x = 2;
    CHECK(p1->x == 3);
  }
}
