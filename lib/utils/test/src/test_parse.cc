#include "doctest.h"
#include "utils/parse.h"

using namespace FlexFlow;

TEST_CASE("Test ArgsParser basic functionality") {
  char const *test_argv[] = {"program_name",
                             "--batch-size",
                             "100",
                             "--learning-rate",
                             "0.5",
                             "--fusion",
                             "true",
                             "-ll:gpus",
                             "6"};
  ArgsParser args;
  auto batch_size_ref =
      add_argument(args, "--batch-size", 32, "batch size for training");
  auto learning_rate_ref = add_argument(
      args, "--learning-rate", 0.01f, "Learning rate for the optimizer");
  auto fusion_ref =
      add_argument(args,
                   "--fusion",
                   "yes",
                   "Flag to determine if fusion optimization should be used");
  auto ll_gpus_ref = add_argument<int>(
      args,
      "-ll:gpus",
      std::nullopt,
      "Number of GPUs to be used for training"); // support non-default value
  parse_args(args, 9, const_cast<char **>(test_argv));

  CHECK(get(args, batch_size_ref) == 100);

  CHECK(get(args, learning_rate_ref) == 0.5f);
  CHECK(get(args, fusion_ref) == true);
  CHECK(get(args, ll_gpus_ref) == 6);
}

TEST_CASE("Test invald command") {
  char const *test_argv[] = {"program_name", "batch-size", "100"};
  ArgsParser args;
  auto batch_size_ref =
      add_argument(args, "batch-size", 32, "batch size for training");
  parse_args(args, 3, const_cast<char **>(test_argv));

  CHECK_THROWS(
      get(args, batch_size_ref)); // throw exception  because we pass batch_size
                                  // via command, it should pass --batch_size
}

TEST_CASE("Test invalid ref") {
  CmdlineArgRef<int> invalid_ref{"invalid", {}};
  char const *test_argv[] = {"program_name"};

  ArgsParser args;
  parse_args(args, 1, const_cast<char **>(test_argv));
  CHECK_THROWS(
      get(args, invalid_ref)); // throw exception  because it's invalid ref
}
