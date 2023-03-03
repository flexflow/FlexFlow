#include "flexflow/utils/dot/record_formatter.h"
#include "gtest/gtest.h"

TEST(record_formatters, basic) {
  RecordFormatter rf, rf2, rf3;
  std::ostringstream oss;
  oss << "Wo"
      << "rld";
  rf << "Hello"
     << "World"
     << (rf2 << "Inner"
             << "World"
             << (rf3 << "Even"
                     << "More"
                     << "Inner World"))
     << "Goodbye" << oss;

  std::ostringstream oss_final;
  oss_final << rf;
  EXPECT_EQ(oss_final.str(),
            "{ Hello | World | { Inner | World | { Even | More | Inner World } "
            "} | Goodbye | World }");
}
