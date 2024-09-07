#include "utils/cli/cli_spec.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cli_parse(CLISpec, std::vector<std::string>)") {
    CLISpec cli = CLISpec{
        {
            CLIFlagSpec{
                "flag-1",
                'f',
                std::nullopt,
            },
            CLIFlagSpec{
                "flag-2",
                std::nullopt,
                "a description of this flag",
            },
        },
        {
            CLIPositionalArgumentSpec{
                "pos-arg-1",
                std::nullopt,
                std::nullopt,
            },
            CLIPositionalArgumentSpec{
                "pos-arg-2",
                std::nullopt,
                "a description of this argument",
            },
        },
    };

    std::vector<std::string> args = {
        "program_name",
        "-f",
        "arg1",
        "arg2",
    };

    tl::expected<CLIParseResult, std::string> result = cli_parse(cli, args);
    tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
        {
            {CLIFlagKey{0}, true},
            {CLIFlagKey{1}, false},
        },
        {
            {CLIPositionalArgumentKey{0}, "arg1"},
            {CLIPositionalArgumentKey{1}, "arg2"},
        },
    };

    CHECK(result == correct);
  }
}
