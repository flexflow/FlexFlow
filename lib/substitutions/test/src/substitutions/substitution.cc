#include <doctest/doctest.h>
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/substitution.h"
#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_valid_substitution") {
    PCGPattern pattern = [] {
      PCGPatternBuilder b;
      PatternValue i1 = b.add_input();
      PatternValue i2 = b.add_input();

      PatternValue o1 = b.add_operator(
        OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR), 
        }},
        {i1, i2},
        {tensor_attribute_pattern_match_all()}
      );


      return b.get_pattern();
    }();

    FAIL("TODO");
  }

  TEST_CASE("apply_substitution") {
    FAIL("TODO");
  }
}
