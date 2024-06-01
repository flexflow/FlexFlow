#include "doctest.h"
#include "utils/parse.h"
#include "utils/tuple.h"

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
  auto batch_size_ref = add_optional_argument(
      args, "--batch-size", optional<int>(32), "batch size for training");
  auto learning_rate_ref =
      add_optional_argument(args,
                            "--learning-rate",
                            optional<float>(0.01f),
                            "Learning rate for the optimizer");
  auto fusion_ref = add_optional_argument(
      args,
      "--fusion",
      optional<bool>("yes"),
      "Flag to determine if fusion optimization should be used");
  auto ll_gpus_ref = add_required_argument<int>(
      args,
      "-ll:gpus",
      std::nullopt,
      "Number of GPUs to be used for training"); // support non-default value
  ArgsParser result = parse_args(args, 9, const_cast<char const **>(test_argv));

  CHECK(get(result, batch_size_ref) == 100);
  CHECK(get(result, learning_rate_ref) == 0.5f);
  CHECK(get(result, fusion_ref) == true);
  CHECK(get(result, ll_gpus_ref) == 6);
}

TEST_CASE("Test invald command") {
  char const *test_argv[] = {"program_name", "batch-size", "100"};
  ArgsParser args;
  auto batch_size_ref = add_optional_argument(
      args, "batch-size", optional<int>(32), "batch size for training");
  CHECK_THROWS(parse_args(
      args,
      3,
      const_cast<char const **>(
          test_argv))); // throw exception  because we pass batch_size  via
                        // command, it should pass --batch_size
}

TEST_CASE("Test invalid ref") {
  CmdlineArgRef<int> invalid_ref{"invalid", {}};
  char const *test_argv[] = {"program_name"};

  ArgsParser args;
  parse_args(args, 1, const_cast<char const **>(test_argv));
  CHECK_THROWS(
      get(args, invalid_ref)); // throw exception  because it's invalid ref
}

TEST_CASE("do not pass the optional argument via command") {
  char const *test_argv[] = {"program_name", "--batch-size", "100"};
  ArgsParser args;
  auto batch_size_ref = add_optional_argument(
      args, "--batch-size", optional<int>(32), "batch size for training");
  auto ll_gpus_ref = add_required_argument<int>(
      args,
      "-ll:gpus",
      std::nullopt,
      "Number of GPUs to be used for training"); // support non-default value
  constexpr size_t test_argv_length = sizeof(test_argv) / sizeof(test_argv[0]);
  CHECK_THROWS(
      parse_args(args, test_argv_length, const_cast<char const **>(test_argv)));
  // throw exception because we don't pass -ll:gpus via command
}

//./a.out --args 4  --arg2 -args3 5  or ./a.out --args 4  --arg2 4 -args3  will
// throw exception
TEST_CASE("only pass the args but not value") {
  SUBCASE("./a.out --args1 4  --arg2 4 -args3  ") {
    char const *test_argv[] = {"program_name",
                               "--batch-size",
                               "100",
                               "--learning-rate",
                               "0.03",
                               "--epoch"};
    ArgsParser args;
    auto batch_size_ref =
        add_optional_argument(args,
                              "--batch-size",
                              std::optional<int>(32),
                              "Size of each batch during training");
    auto learning_rate_ref =
        add_optional_argument(args,
                              "--learning-rate",
                              std::optional<float>(0.001),
                              "Learning rate for the optimizer");
    auto epoch_ref = add_optional_argument(args,
                                           "--epoch",
                                           std::optional<int>(1),
                                           "Number of epochs for training");
    constexpr size_t test_argv_length =
        sizeof(test_argv) / sizeof(test_argv[0]);
    CHECK_THROWS(parse_args(
        args, test_argv_length, const_cast<char const **>(test_argv)));
  }

  SUBCASE("./a.out --args 4  --arg2 -args3 4") {
    char const *test_argv[] = {
        "program_name",
        "--batch-size",
        "100",
        "--epoch",
        "--learning-rate",
        "0.03",
    };
    ArgsParser args;
    auto batch_size_ref =
        add_optional_argument(args,
                              "--batch-size",
                              std::optional<int>(32),
                              "Size of each batch during training");
    auto learning_rate_ref =
        add_optional_argument(args,
                              "--learning-rate",
                              std::optional<float>(0.001),
                              "Learning rate for the optimizer");
    auto epoch_ref = add_optional_argument(args,
                                           "--epoch",
                                           std::optional<int>(1),
                                           "Number of epochs for training");
    constexpr size_t test_argv_length =
        sizeof(test_argv) / sizeof(test_argv[0]);
    CHECK_THROWS(parse_args(
        args, test_argv_length, const_cast<char const **>(test_argv)));
  }
}

TEST_CASE("support action_true") {

  ArgsParser args;
  auto verbose_ref = add_optional_argument(args,
                                           "--verbose",
                                           std::optional<bool>(false),
                                           "Whether to print verbose logs",
                                           true);
  SUBCASE("do not pass --verbose via command") {
    constexpr size_t test_argv_length =
        sizeof(test_argv) / sizeof(test_argv[0]);
    char const *test_argv[] = {"program_name"};
    ArgsParser result = parse_args(
        args, test_argv_length, const_cast<char const **>(test_argv));
    CHECK(get(result, verbose_ref) == false);
  }

  SUBCASE("pass --verbose via command") {
    constexpr size_t test_argv_length =
        sizeof(test_argv) / sizeof(test_argv[0]);
    char const *test_argv[] = {"program_name", "--verbose"};
    ArgsParser result = parse_args(
        args, test_argv_length, const_cast<char const **>(test_argv));
    CHECK(get(result, verbose_ref) == true);
  }
}
