#include "utils/testing.h"
#include "utils/smart_ptrs/cow_ptr_t.h"

struct clonable_t {
  clonable_t *clone() const {
    return new clonable_t{};
  }
};

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
    auto p1 = make_cow_ptr<TestObject>(1);
    auto p2 = make_cow_ptr<TestObject>(2);
    p1 = p2;
    CHECK(p1->x == p2->x);
  }

  TEST_CASE("cow_ptr_t cast") {
    auto p1 = make_cow_ptr<TestObjectDerived>(1, 2);
    cow_ptr_t<TestObject> p2 = p1;
    CHECK(p2->x == 1);
  }

  TEST_CASE("cow_ptr_t mutation under exclusive ownership") {
    auto p1 = make_cow_ptr<TestObject>(1);
    TestObject const *orig_addr = p1.get_raw_unsafe();

    p1.get_mutable()->x = 2;
    CHECK(p1->x == 2);
    CHECK(p1.get_raw_unsafe() == orig_addr);
  }

  TEST_CASE("cow_ptr_t mutation under shared ownership") {
    auto p1 = make_cow_ptr<TestObject>(1);
    TestObject const *orig_addr = p1.get_raw_unsafe();
    auto p2 = p1;
    CHECK(p2.get_raw_unsafe() == orig_addr);

    p1.get_mutable()->x = 2;
    CHECK(p1->x == 2);
    CHECK(p2->x == 1);
    CHECK(p2.get_raw_unsafe() == orig_addr);
    CHECK(p1.get_raw_unsafe() != orig_addr);

    p2.get_mutable()->x = 3;
    CHECK(p1->x == 2);
    CHECK(p2->x == 3);
    CHECK(p2.get_raw_unsafe() == orig_addr);
  }

  TEST_CASE("cow_ptr_t mutation with type change") {
    auto p1 = make_cow_ptr<TestObjectDerived>(1, 2);
    cow_ptr_t<TestObject> p2 = p1;
    p2.get_mutable()->x = 3;
    CHECK(reinterpret_pointer_cast<TestObjectDerived>(p2)->y == 2);
  }

  TEST_CASE("reinterpret_pointer_cast") {
    auto p1 = make_cow_ptr<TestObjectDerived>(1, 2);
    cow_ptr_t<TestObject> p2 = p1;
    reinterpret_pointer_cast<TestObjectDerived>(p2).get_mutable()->y = 4;
    CHECK(p1->y == 2);
  }

  TEST_CASE("cow_ptr_t get_mutable") {
    auto p1 = make_cow_ptr<TestObject>(1);
    cow_ptr_t<TestObject> p2 = p1;
    p1.get_mutable()->x = 3;
    CHECK(p1->x == 3);
    CHECK(p2->x == 1);
    p2.get_mutable()->x = 2;
    CHECK(p1->x == 3);
  }

  TEST_CASE("cow_ptr_t") {
    CHECK(make_cow_ptr<clonable_t>().get().get() != nullptr);
  }
}
