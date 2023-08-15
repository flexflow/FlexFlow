#include "doctest.h"
#include "kernels/datatype_dispatch.h"

using namespace FlexFlow;

struct TestFn {
    template <DataType DT>
    int operator()(int x) const {
        if (DT == DataType::FLOAT) return x + 1;
        else if (DT == DataType::DOUBLE) return x + 2;
        // 添加其他类型的逻辑
        return 0;
    }
};

TEST_CASE("dispatch") {
    SUBCASE("Dispatch float") {
        CHECK(dispatch<TestFn>(DataType::FLOAT, 5) == 6);
    }

    SUBCASE("Dispatch double") {
        CHECK(dispatch<TestFn>(DataType::DOUBLE, 5) == 7);
    }
}

TEST_CASE("DataTypeDispatch1") {
    DataTypeDispatch1<TestFn> dispatcher;

    SUBCASE("Dispatch1 float") {
        CHECK(dispatcher(DataType::FLOAT, 5) == 6);
    }

    SUBCASE("Dispatch1 double") {
        CHECK(dispatcher(DataType::DOUBLE, 5) == 7);
    }
}

TEST_CASE("DataTypeDispatch2") {
    DataTypeDispatch2<TestFn> dispatcher;

    SUBCASE("Dispatch2 float") {
        CHECK(dispatcher(DataType::FLOAT, 5) == 6);
    }

    SUBCASE("Dispatch2 double") {
        CHECK(dispatcher(DataType::DOUBLE, 5) == 7);
    }
}
