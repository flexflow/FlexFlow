#include "doctest.h"
#include "kernels/datatype_dispatch.h"

using namespace FlexFlow;

template <DataType DT>
struct Function1 {
  int operator()(int value) const {
    if (DT == DataType::FLOAT) {
      return value + 1;
    }
    if (DT == DataType::DOUBLE) {
      return value + 2;
    }
    return 0;
  }
};

TEST_CASE("Testing dispatch function") {
  int value = 10;
  int result = dispatch<Function1>(DataType::FLOAT, value);
  CHECK(result == 11);
}

// test DataTypeDispatch1
TEST_CASE("Testing DataTypeDispatch1") {
  DataTypeDispatch1<Function1> dispatcher;
  int value = 10;
  int result = dispatcher(DataType::FLOAT, value);
  CHECK(result == 11);
}
