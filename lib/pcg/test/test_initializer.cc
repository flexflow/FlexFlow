#include "doctest.h"
#include "pcg/file_format/v1/initializer.h"
#include "utils.h"
#include "utils/containers.h"
#include "utils/required.h"

using namespace FlexFlow;

TEST_CASE("GlorotInitializer") {
  GlorotUniform i = {11};
  V1GlorotInitializer v1 = to_v1(i);

  // CHECK(from_v1(v1) == i);

  json j = v1;
  check_fields(j, {{"seed", "11"}});
}

TEST_CASE("ZeroInitializer") {
  ZeroInitializer i;
  V1ZeroInitializer v1 = to_v1(i);

  // CHECK(from_v1(v1) == i);

  json j = v1;
  check_fields(j, {});
}

TEST_CASE("UniformInitializer") {
  UniformInitializer i = {77, 9.1, 4.3};
  V1UniformInitializer v1 = to_v1(i);

  // CHECK(from_v1(v1) == i);

  json j = v1;
  check_fields(j, {{"seed", "77"}, {"min_val", "9.1"}, {"max_val", "4.3"}});
}

TEST_CASE("NormInitializer") {
  NormInitializer i = {77, 9.1, 4.3};
  V1NormInitializer v1 = to_v1(i);

  // CHECK(from_v1(v1) == i);

  json j = v1;
  check_fields(j, {{"seed", "77"}, {"mean", "9.1"}, {"stddev", "4.3"}});
}

TEST_CASE("ConstantInitializer") {
  ConstantInitializer i = {32};
  V1ConstantInitializer v1 = to_v1(i);

  // CHECK(from_v1(v1) == i);

  json j = v1;
  // The value field is a variant. Don't try to check for anything in the
  // serialization because that will have been tested elsewhere. Just check that
  // the value of "value" is an object which is good enough.
  check_fields(j, {{"value", "{"}});
}

TEST_CASE("Initializer") {
  Initializer ig = GlorotUniform{11};
  V1Initializer v1g = to_v1(ig);
  // CHECK(from_v1(v1g) == ig);

  Initializer iz = ZeroInitializer{};
  V1Initializer v1z = to_v1(iz);
  // CHECK(from_v1(v1z) == iz);

  Initializer iu = UniformInitializer{77, 9.1, 4.3};
  V1Initializer v1u = to_v1(iu);
  // CHECK(from_v1(v1u) == iu);

  Initializer in = NormInitializer{77, 9.1, 4.3};
  V1Initializer v1n = to_v1(in);
  // CHECK(from_v1(v1n) == in);

  Initializer ic = ConstantInitializer{32};
  V1Initializer v1c = to_v1(ic);
  // CHECK(from_v1(v1c) == ic);

  // No need to check the JSON because Initializer is just a variant.
}
