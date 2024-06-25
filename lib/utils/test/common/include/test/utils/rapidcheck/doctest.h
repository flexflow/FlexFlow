#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_DOCTEST_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_DOCTEST_H

#include "doctest/doctest.h"
#include "rapidcheck.h"

namespace FlexFlow {

template <class testable>
void RC_SUBCASE(char const *d, testable &&t, bool v = false) {
  using namespace ::rc;
  using namespace ::rc::detail;
  using namespace ::doctest::detail;

  DOCTEST_SUBCASE(d) {
    auto r = checkTestable(std::forward<testable>(t));

    if (r.template is<SuccessResult>()) {
      if (!r.template get<SuccessResult>().distribution.empty() || v) {
        std::cout << "- " << d << std::endl;
        printResultMessage(r, std::cout);
        std::cout << std::endl;
      }

      REQUIRE(true);
    } else {
      std::ostringstream o;
      printResultMessage(r, o << '\n');
      DOCTEST_INFO(o.str());
      ResultBuilder b(doctest::assertType::DT_CHECK, __FILE__, __LINE__, d);
      DOCTEST_ASSERT_LOG_REACT_RETURN(b);
    }
  }
}
template <class testable>
void RC_SUBCASE(testable &&t, bool v = false) {
  RC_SUBCASE("", t, v);
}

} // namespace FlexFlow

#endif
