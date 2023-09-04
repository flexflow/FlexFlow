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
  auto batch_size_ref = args.add_argument(
      "--batch-size", 32, "Size of each batch during training");
  auto learning_rate_ref = args.add_argument(
      "--learning-rate", 0.01f, "Learning rate for the optimizer");
  auto fusion_ref = args.add_argument(
      "--fusion",
      false,
      "Flag to determine if fusion optimization should be used");
  auto ll_gpus_ref = args.add_argument(
      "-ll:gpus", 2, "Number of GPUs to be used for training");
  args.parse_args(9, const_cast<char **>(test_argv));

  CHECK(args.get(batch_size_ref) == 100);
  CHECK(args.get(learning_rate_ref) == 0.5f);
  CHECK(args.get(fusion_ref) == true);
  CHECK(args.get(ll_gpus_ref) == 6);
}
