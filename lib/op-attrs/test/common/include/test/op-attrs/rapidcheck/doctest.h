#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_DOCTEST_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_DOCTEST_H

#include "doctest/doctest.h"
#include "rapidcheck.h"

namespace rc {

/**
 * Checks the given predicate by applying it to randomly generated arguments.
 *
 * Quotes the given description string if the predicate can be falsified.
 *
 * Traces a progress message to 'stdout' if the flag 'v' is true.
 *
 * Like the function 'rc::check', but integrates with 'doctest' to include its
 * result in the statistics that are gathered for a test run.
 *
 * For example:
 *
 *  TEST_CASE("addition is commutative")
 *  {
 *    wol::test::check("a+b == b+a", [](int a, int b) { return a+b == b+a; });
 *  }
 *
 * @param  d  A description of the predicate being checked.
 * @param  t  A predicate to check.
 * @param  v  A flag requesting verbose output.
 *
 * @see    https://github.com/emil-e/rapidcheck/blob/master/doc/properties.md
 *         for more on 'rc::check', on which this function is modeled.
 *
 * @see    https://github.com/emil-e/rapidcheck/blob/master/doc/catch.md
 *         for more on the integration of 'rapidcheck' and 'catch', on which
 *         this implementation is based.
 */
template <class testable>
void dc_check(char const *d, testable &&t, bool v = false) {
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

/**
 * Checks the given predicate by applying it to randomly generated arguments.
 *
 * Quotes the given description string if the predicate can be falsified.
 *
 * Traces a progress message to 'stdout' if the flag 'v' is true.
 *
 * Like the function 'rc::check', but integrates with 'doctest' to include its
 * result in the statitics that are gathered for a test run.
 *
 * For example:
 *
 *  TEST_CASE("addition is commutative")
 *  {
 *    wol::test::check("a+b == b+a", [](int a, int b) { return a+b == b+a; });
 *  }
 *
 * @param  t  A predicate to check.
 * @param  v  A flag requesting verbose output.
 *
 * @see    https://github.com/emil-e/rapidcheck/blob/master/doc/properties.md
 *         for more on 'rc::check', on which this function is modeled.
 *
 * @see    https://github.com/emil-e/rapidcheck/blob/master/doc/catch.md
 *         for more on the integration of 'rapidcheck' and 'catch', on which
 *         this implementation is based.
 */
template <class testable>
inline void dc_check(testable &&t, bool v = false) {
  check("", t, v);
}

#define RC_SUBCASE(NAME) rc

} // namespace rc

#endif
