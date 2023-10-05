#include "doctest.h"
#include "pcg/file_format/v1/tensor.h"
#include "pcg/tensor.h"
#include "utils/containers.h"
#include "utils/json.h"
#include "utils/required.h"

#include <optional>

using namespace FlexFlow;

TEST_CASE("Tensor") {
  Tensor t{{3, 4, 5},
           DataType::FLOAT,
           CreateGrad::NO,
           nullopt,
           nullopt,
           std::string("tensor")};
  json j = to_v1(t);

  // FIXME: There is currently a bug in the serialization of tl::optional to
  // json which puts it in an infinite loop. When that is fixed, this will
  // actually work.
  NOT_REACHABLE();
}
