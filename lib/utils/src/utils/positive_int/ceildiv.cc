#include "utils/positive_int/ceildiv.h"

namespace FlexFlow {

positive_int ceildiv(positive_int numerator, positive_int denominator) {
  int n = numerator.int_from_positive_int();
  int d = denominator.int_from_positive_int();

  int result = (n + d - 1) / d;
  return positive_int{result};
}

} // namespace FlexFlow
