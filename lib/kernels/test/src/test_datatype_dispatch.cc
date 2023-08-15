// #include "doctest.h"
// #include "rapidcheck.h"

// using namespace FlexFlow;

// struct TestFn {
//   template <DataType DT>
//   int operator()(int x) const {
//     if (DT == DataType::FLOAT) {
//       return x + 1;
//     } else if (DT == DataType::DOUBLE) {
//       return x + 2;
//     }
//     return 0;
//   }
// };

// RC_GTEST_PROP(dispatch, "Dispatch float", (int x)) {
//   RC_ASSERT(dispatch<TestFn>(DataType::FLOAT, x) == x + 1);
// }

// RC_GTEST_PROP(dispatch, "Dispatch double", (int x)) {
//   RC_ASSERT(dispatch<TestFn>(DataType::DOUBLE, x) == x + 2);
// }

// RC_GTEST_PROP(DataTypeDispatch1, "Dispatch1 float", (int x)) {
//   DataTypeDispatch1<TestFn> dispatcher;
//   RC_ASSERT(dispatcher(DataType::FLOAT, x) == x + 1);
// }

// RC_GTEST_PROP(DataTypeDispatch1, "Dispatch1 double", (int x)) {
//   DataTypeDispatch1<TestFn> dispatcher;
//   RC_ASSERT(dispatcher(DataType::DOUBLE, x) == x + 2);
// }

// RC_GTEST_PROP(DataTypeDispatch2, "Dispatch2 float", (int x)) {
//   DataTypeDispatch2<TestFn> dispatcher;
//   RC_ASSERT(dispatcher(DataType::FLOAT, x) == x + 1);
// }

// RC_GTEST_PROP(DataTypeDispatch2, "Dispatch2 double", (int x)) {
//   DataTypeDispatch2<TestFn> dispatcher;
//   RC_ASSERT(dispatcher(DataType::DOUBLE, x) == x + 2);
// }
