#include "doctest.h"
#include "kernels/legion_dim.h"

using namespace FlexFlow;

TEST_CASE("Testing DimOrdered") {
    SUBCASE("constructor method") {
        DimOrdered<legion_dim_t, int> fromInitList = {1, 2, 3};
        CHECK(fromInitList.size() == 3);
        std::vector<int> vec = {4, 5, 6};
        DimOrdered<legion_dim_t, int> fromVector(vec);
        CHECK(fromVector.size() == 3);
    }

    SUBCASE("at") {
        DimOrdered<legion_dim_t, int> dimOrder = {1, 2, 3};
        CHECK(dimOrder[legion_dim_t(0)] == 1);
        CHECK(dimOrder[legion_dim_t(1)] == 2);
        CHECK(dimOrder[legion_dim_t(2)] == 3);
    }

    SUBCASE("comparsion") {
        DimOrdered<legion_dim_t, int> order1 = {1, 2, 3};
        DimOrdered<legion_dim_t, int> order2 = {1, 2, 4};
        DimOrdered<legion_dim_t, int> order3 = {1, 2, 3};
        
        CHECK(order1 != order2);
        CHECK(order1 == order3);
    }

    SUBCASE("iterator") {
        DimOrdered<legion_dim_t, int> dimOrder = {1, 2, 3};
        int sum = 0;
        for (int value : dimOrder) {
            sum += value;
        }
        CHECK(sum == 6);
    }

}

TEST_CASE("Testing LegionTensorDims") {

    SUBCASE("LegionTensorDims Basic Operation") {
        LegionTensorDims tensorDims;
        
        tensorDims[legion_dim_t(1)] = 100;
        CHECK(tensorDims[legion_dim_t(1)] == 100);
        
        tensorDims[legion_dim_t(2)] = 200;
        CHECK(tensorDims[legion_dim_t(2)] == 200);
    }
}

