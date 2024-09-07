#include "utils/cli/cli_get_help_message.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cli_get_help_message(std::string, CLISpec)") {
    CLISpec cli = CLISpec{
        {
            CLIFlagSpec{
                "flag-1",
                'f',
            },
            CLIFlagSpec{
                "flag-2",
                std::nullopt,
            },
        },
        {
            CLIPositionalArgumentSpec{
                "pos-arg-1",
                std::nullopt,
            },
            CLIPositionalArgumentSpec{
                "pos-arg-2",
                std::vector<std::string>{"red", "green", "blue"},
            },
        },
    };
    std::string program_name = "prog_name";

    std::string correct = (
      "usage: prog_name [-h] [-f] [--flag-2] pos-arg-1 {red,green,blue}\n"
      "\n"
      "positional arguments:\n"
      "  pos-arg-1\n"
      "  {red,green,blue}\n"
      "\n"
      "options:\n"
      "  -h, --help    show this help message and exit\n"
      "  -f, --flag-1  \n"
      "  --flag-2\n"
    );

    std::string result = cli_get_help_message(program_name, cli);

    CHECK(result == correct);
  }
}
