#include "substitutions/unlabelled/match_additional_criterion.h"

namespace FlexFlow {

MatchAdditionalCriterion match_additional_crition_always_true() {
  return MatchAdditionalCriterion{
      [](PatternNode const &, Node const &) { return true; },
      [](PatternValue const &, OpenDataflowValue const &) { return true; },
  };
}

} // namespace FlexFlow
