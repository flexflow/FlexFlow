#include "doctest.h"
#include "pcg/file_format/v1/parallel_tensor.h"
#include "pcg/parallel_tensor.h"
#include "utils.h"
#include "utils/json.h"
#include "utils/required.h"

using namespace FlexFlow;

TEST_CASE("ParallelDim") {
  ParallelDim d{4, 11, false};
  V1ParallelDim v1 = to_v1(d);

  CHECK(from_v1(v1) == d);

  json j = v1;
  check_fields(j, {{"size", "4"},
                   {"degree", "11"},
                   {"is_replica_dim", "false"}});
  // TODO: Check deserialization.
}

TEST_CASE("ParallelTensorDims") {
  ParallelTensorDims d(std::vector<ParallelDim>{{3, 11, false}, {4, 2, true}});
  V1ParallelTensorDims v1 = to_v1(d);

  CHECK(from_v1(v1) == d);

  json j = v1;

  // Currently, there isn't a great way to check the actual JSON since the
  // elements of the array will also be objects whose fields may be in a
  // different order. The test utilities are ok for checking that the fields of
  // an object were serialized but they are currently not up to checking if
  // an array of objects was serialized. So just check that it is an array of
  // objects that was serialized.
  std::string strj = str(j);
  CHECK(strj.find("{\"data\":[{") == 0);
  CHECK(strj.substr(strj.size() - 3, 3) == "}]}");

  // TODO: Check deserialization.
}

TEST_CASE("ParallelTensorShape") {
  ParallelTensorShape t{std::vector<ParallelDim>{{3, 11, false}, {4, 2, true}},
                        DataType::FLOAT};
  V1ParallelTensorShape v1 = to_v1(t);

  CHECK(from_v1(v1) == t);

  json j = v1;

  // We can't check the serialization of the dims (see comment in the test case
  // for ParallelTensorDims)
  check_fields(j, {{"dims", "{\"data\":[{"}, {"data_type", "\"FLOAT\""}});

  // TODO: Check deserialization.
}

TEST_CASE("ParallelTensor") {
  ParallelTensor t{std::vector<ParallelDim>{{3, 11, false}, {4, 2, true}},
                   DataType::FLOAT,
                   CreateGrad::NO,
                   GlorotUniform{932},
                   ParamSync::PS,
                   std::string("tensor")};
  V1ParallelTensor v1 = to_v1(t);

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

  ParallelTensor t0{ParallelTensorDims(
                        std::vector<ParallelDim>{{3, 11, false}, {4, 2, true}}),
                    DataType::FLOAT,
                    CreateGrad::YES,
                    nullopt,
                    nullopt,
                    nullopt};
  V1ParallelTensor v10 = to_v1(t0);

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
