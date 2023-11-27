#include "doctest.h"
#include "pcg/file_format/v1/tensor.h"
#include "pcg/tensor.h"
#include "utils.h"
#include "utils/json.h"
#include "utils/required.h"

using namespace FlexFlow;

TEST_CASE("TensorShape") {
  TensorShape t{{3, 4, 5}, DataType::FLOAT};
  V1TensorShape v1 = to_v1(t);

  CHECK(from_v1(v1) == t);

  json j = v1;
  check_fields(j, {{"dims", "[3,4,5]"}, {"data_type", "\"FLOAT\""}});
  // TODO: Check deserialization.
}

TEST_CASE("Tensor") {
  Tensor t{{3, 4, 5},
           DataType::FLOAT,
           CreateGrad::NO,
           GlorotUniform{932},
           ParamSync::PS,
           std::string("tensor")};
  V1Tensor v1 = to_v1(t);

  CHECK(from_v1(v1) == t);

  json j = v1;
  // shape is itself an object. Since the order of the fields there may not be
  // consistent, just check that the key exists. This is particularly relevant
  // since there is no field named shape in the tensor object.
  check_fields(j,
               {{"shape", "{"},
                {"create_gradients", "\"NO\""},
                {"initializer", "{"},
                {"sync_type", "\"PARAM_SERVER\""},
                {"name", "\"tensor\""}});
  // TODO: Check deserialization.

  Tensor t0{
      {3, 4, 5}, DataType::FLOAT, CreateGrad::YES, nullopt, nullopt, nullopt};
  V1Tensor v10 = to_v1(t0);

  CHECK(from_v1(v10) == t0);

  json j0 = v10;
  check_fields(j0,
               {{"shape", "{"},
                {"create_gradients", "\"YES\""},
                {"initializer", "null"},
                {"sync_type", "null"},
                {"name", "null"}});
  // TODO: Check deserialization.
}
