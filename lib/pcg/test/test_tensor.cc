#include "doctest.h"
#include "pcg/file_format/v1/tensor.h"
#include "pcg/tensor.h"
#include "utils/containers.h"
#include "utils/required.h"

#include <optional>

using namespace FlexFlow;

TEST_CASE("Tensor") {
  Tensor t{{3, 4, 5},
           DataType::FLOAT,
           false,
           nullopt,
           nullopt,
           std::string("tensor")};
  to_v1(t);
}
