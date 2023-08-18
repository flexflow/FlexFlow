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
  args.add_argument("--batch-size", 32, "Size of each batch during training");
  args.add_argument(
      "--learning-rate", 0.01f, "Learning rate for the optimizer");
  args.add_argument("--fusion",
                    false,
                    "Flag to determine if fusion optimization should be used");
  args.add_argument("-ll:gpus", 2, "Number of GPUs to be used for training");
  args.parse_args(9, const_cast<char **>(test_argv));

  CHECK(args.get<int>("batch-size") == 100);
  CHECK(args.get<float>("learning-rate") == 0.5f);
  CHECK(args.get<bool>("fusion") == true);
  CHECK(args.get<int>("-ll:gpus") == 6);
}
