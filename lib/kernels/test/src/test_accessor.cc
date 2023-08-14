
#include "doctest.h"
#include "kernels/accessor.h"

using namespace FlexFlow;

TEST_CASE("Test GenericTensorAccessorW") {
    GenericTensorAccessorW accessorW;
    float dataW = 3.14f;
    accessorW.data_type = DataType::FLOAT;
    accessorW.ptr = &dataW;

    // Test get method
    CHECK(*accessorW.get<DataType::FLOAT>() == doctest::Approx(3.14f));

    // Test specific type get ptr methods
    CHECK(accessorW.get_float_ptr() != nullptr);
    CHECK(*accessorW.get_float_ptr() == doctest::Approx(3.14f));

    // Check runtime error for invalid access
    CHECK_THROWS_WITH(accessorW.get<DataType::DOUBLE>(), 
                      "Invalid access data type (FLOAT != DOUBLE)");
}

TEST_CASE("Test GenericTensorAccessorR") {
    GenericTensorAccessorR accessorR;
    float dataR = 7.89f;
    accessorR.data_type = DataType::FLOAT;
    accessorR.ptr = &dataR;

    // Test get method
    CHECK(*accessorR.get<DataType::FLOAT>() == doctest::Approx(7.89f));

    // Test specific type get ptr methods
    CHECK(accessorR.get_float_ptr() != nullptr);
    CHECK(*accessorR.get_float_ptr() == doctest::Approx(7.89f));

    // Check runtime error for invalid access
    CHECK_THROWS_WITH(accessorR.get<DataType::DOUBLE>(), 
                      "Invalid access data type (FLOAT != DOUBLE)");
}

TEST_CASE("Test get_int32_ptr for GenericTensorAccessorW") {
    GenericTensorAccessorW accessorW;
    int32_t dataW = 12345;
    accessorW.data_type = DataType::INT32;
    accessorW.ptr = &dataW;

    // Test get_int32_ptr method
    CHECK(accessorW.get_int32_ptr() != nullptr);
    CHECK(*accessorW.get_int32_ptr() == 12345);
}

TEST_CASE("Test get_int64_ptr for GenericTensorAccessorW") {
    GenericTensorAccessorW accessorW;
    int64_t dataW = 1234567890LL;
    accessorW.data_type = DataType::INT64;
    accessorW.ptr = &dataW;

    // Test get_int64_ptr method
    CHECK(accessorW.get_int64_ptr() != nullptr);
    CHECK(*accessorW.get_int64_ptr() == 1234567890LL);
}

TEST_CASE("Test get_float_ptr for GenericTensorAccessorW") {
    GenericTensorAccessorW accessorW;
    float dataW = 3.14f;
    accessorW.data_type = DataType::FLOAT;
    accessorW.ptr = &dataW;

    // Test get_float_ptr method
    CHECK(accessorW.get_float_ptr() != nullptr);
    CHECK(*accessorW.get_float_ptr() == doctest::Approx(3.14f));
}

TEST_CASE("Test get_double_ptr for GenericTensorAccessorW") {
    GenericTensorAccessorW accessorW;
    double dataW = 6.28;
    accessorW.data_type = DataType::DOUBLE;
    accessorW.ptr = &dataW;

    // Test get_double_ptr method
    CHECK(accessorW.get_double_ptr() != nullptr);
    CHECK(*accessorW.get_double_ptr() == doctest::Approx(6.28));
}

// You can repeat similar tests for GenericTensorAccessorR

TEST_CASE("Test get_int32_ptr for GenericTensorAccessorR") {
    GenericTensorAccessorR accessorR;
    int32_t dataR = 67890;
    accessorR.data_type = DataType::INT32;
    accessorR.ptr = &dataR;

    // Test get_int32_ptr method
    CHECK(accessorR.get_int32_ptr() != nullptr);
    CHECK(*accessorR.get_int32_ptr() == 67890);
}
