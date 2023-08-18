
#include "doctest.h"
#include "kernels/accessor.h"

using namespace FlexFlow;

TEST_CASE("Test GenericTensorAccessorW") {
  float dataW = 3.14f;
  GenericTensorAccessorW accessorW{
      DataType::FLOAT, ArrayShape(std::vector<std::size_t>{}), &dataW};

  // Test get method
  CHECK(*accessorW.get<DataType::FLOAT>() == doctest::Approx(3.14f));

  // Test specific type get ptr methods
  CHECK(get_float_ptr(accessorW) != nullptr);
  CHECK(*get_float_ptr(accessorW) == doctest::Approx(3.14f));

  // Check runtime error for invalid access
  CHECK_THROWS_WITH(accessorW.get<DataType::DOUBLE>(),
                    "Invalid access data type (FLOAT != DOUBLE)");
}

TEST_CASE("Test GenericTensorAccessorR") {
  float dataR = 7.89f;
  GenericTensorAccessorR accessorR{
      DataType::FLOAT, ArrayShape(std::vector<std::size_t>{}), &dataR};
  // Test get method
  CHECK(*accessorR.get<DataType::FLOAT>() == doctest::Approx(7.89f));

  // Test specific type get ptr methods
  CHECK(get_float_ptr(accessorR) != nullptr);
  CHECK(*get_float_ptr(accessorR) == doctest::Approx(7.89f));

  // Check runtime error for invalid access
  CHECK_THROWS_WITH(accessorR.get<DataType::DOUBLE>(),
                    "Invalid access data type (FLOAT != DOUBLE)");
}

TEST_CASE("Test get_int32_ptr for GenericTensorAccessorW") {
  int32_t dataW = 12345;
  GenericTensorAccessorW accessorW{
      DataType::INT32, ArrayShape(std::vector<std::size_t>{}), &dataW};

  // Test get_int32_ptr method
  CHECK(get_int32_ptr(accessorW) != nullptr);
  CHECK(*get_int32_ptr(accessorW) == 12345);
}

TEST_CASE("Test get_int64_ptr for GenericTensorAccessorW") {
  int64_t dataW = 1234567890LL;
  GenericTensorAccessorW accessorW{
      DataType::INT64, ArrayShape(std::vector<std::size_t>{}), &dataW};
  // Test get_int64_ptr method
  CHECK(get_int64_ptr(accessorW) != nullptr);
  CHECK(*get_int64_ptr(accessorW) == 1234567890LL);
}

TEST_CASE("Test get_float_ptr for GenericTensorAccessorW") {
  float dataW = 3.14f;
  GenericTensorAccessorW accessorW{
      DataType::FLOAT, ArrayShape(std::vector<std::size_t>{}), &dataW};
  // Test get_float_ptr method
  CHECK(get_float_ptr(accessorW) != nullptr);
  CHECK(*get_float_ptr(accessorW) == doctest::Approx(3.14f));
}

TEST_CASE("Test get_double_ptr for GenericTensorAccessorW") {
  double dataW = 6.28;
  GenericTensorAccessorW accessorW{
      DataType::DOUBLE, ArrayShape(std::vector<std::size_t>{}), &dataW};
  // Test get_double_ptr method
  CHECK(get_double_ptr(accessorW) != nullptr);
  CHECK(*get_double_ptr(accessorW) == doctest::Approx(6.28));
}

TEST_CASE("Test get_int32_ptr for GenericTensorAccessorR") {
  int32_t dataR = 67890;
  GenericTensorAccessorR accessorR{
      DataType::INT32, ArrayShape(std::vector<std::size_t>{}), &dataR};
  // Test get_int32_ptr method
  CHECK(get_int32_ptr(accessorR) != nullptr);
  CHECK(*get_int32_ptr(accessorR) == 67890);
}
