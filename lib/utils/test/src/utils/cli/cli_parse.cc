#include "utils/cli/cli_parse.h"
#include "utils/expected.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cli_parse_flag(CLISpec, std::string)") {
    CLISpec cli = CLISpec{
      {
        CLIFlagSpec{
          "flag1",
          std::nullopt,
          std::nullopt,
        },
        CLIFlagSpec{
          "flag2",
          '2',
          std::nullopt,
        },
      },
      {},
    };

    CLIFlagKey key_flag1 = CLIFlagKey{0};
    CLIFlagKey key_flag2 = CLIFlagKey{1};

    SUBCASE("correctly parses short flag") {
      std::string input = "-2";
      
      tl::expected<CLIFlagKey, std::string> result = cli_parse_flag(cli, input); 
      tl::expected<CLIFlagKey, std::string> correct = key_flag2;

      CHECK(result == correct);
    }

    SUBCASE("correctly parses long flag") {
      std::string input = "--flag1";

      tl::expected<CLIFlagKey, std::string> result = cli_parse_flag(cli, input); 
      tl::expected<CLIFlagKey, std::string> correct = key_flag1;

      CHECK(result == correct);
    }

    SUBCASE("fails on unknown flag") {
      std::string input = "--not-real";

      tl::expected<CLIFlagKey, std::string> result = cli_parse_flag(cli, input);
      tl::expected<CLIFlagKey, std::string> correct = tl::unexpected("Encountered unknown flag --not-real");

      CHECK(result == correct);
    }

    SUBCASE("fails on non-flag") {
      std::string input = "-flag1";

      std::optional<CLIFlagKey> result = optional_from_expected(cli_parse_flag(cli, input));
      std::optional<CLIFlagKey> correct = std::nullopt;

      CHECK(result == correct);
    }
  }

  TEST_CASE("cli_parse(CLISpec, std::vector<std::string>)") {
    SUBCASE("works even if cli is empty") {
      CLISpec cli = CLISpec{{}, {}};
      std::vector<std::string> inputs = {"prog_name"};

      tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
      tl::expected<CLIParseResult, std::string> correct = CLIParseResult{{}, {}};

      CHECK(result == correct);
    }

    SUBCASE("flag parsing") {
      CLISpec cli = CLISpec{
        {
          CLIFlagSpec{
            "flag1",
            std::nullopt,
            std::nullopt,
          },
          CLIFlagSpec{
            "flag2",
            '2',
            std::nullopt,
          },
        },
        {},
      };
      CLIFlagKey key_flag1 = CLIFlagKey{0};
      CLIFlagKey key_flag2 = CLIFlagKey{1};

      SUBCASE("parses flags in any order") {
        std::vector<std::string> inputs = {"prog_name", "-2", "--flag1"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, true},
            {key_flag2, true},
          },
          {},
        };

        CHECK(result == correct);
      }

      SUBCASE("is fine if some are not present") {
        std::vector<std::string> inputs = {"prog_name", "-2"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, false},
            {key_flag2, true},
          },
          {},
        };

        CHECK(result == correct);
      }

      SUBCASE("is fine if none are present") {
        std::vector<std::string> inputs = {"prog_name"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, false},
            {key_flag2, false},
          },
          {},
        };

        CHECK(result == correct);
      }

      SUBCASE("is fine even if the program name is a flag") {
        std::vector<std::string> inputs = {"--flag1", "-2"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, false},
            {key_flag2, true},
          },
          {},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("positional argument parsing") {
      SUBCASE("without choices") {
        CLISpec cli = CLISpec{
          {},
          {
            CLIPositionalArgumentSpec{
              "posarg1",
              std::nullopt,
              std::nullopt,
            },
            CLIPositionalArgumentSpec{
              "posarg2",
              std::nullopt,
              std::nullopt,
            },
          },
        };

        CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0};
        CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1};

        SUBCASE("can parse multiple positional arguments") {
          std::vector<std::string> inputs = {"prog_name", "hello", "world"};

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs); 
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            {},
            {
              {key_posarg1, "hello"},
              {key_posarg2, "world"},
            }
          };

          CHECK(result == correct);
        }

        SUBCASE("requires all positional arguments to be present") {
          std::vector<std::string> inputs = {"prog_name", "hello"};

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs); 
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected("Not enough positional arguments: found 1, expected 2");

          CHECK(result == correct);
        }

        SUBCASE("requires no extra positional arguments to be present") {
          std::vector<std::string> inputs = {"prog_name", "hello", "there", "world" };

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs); 
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected("Too many positional arguments: expected 2");

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to contain spaces") {
          std::vector<std::string> inputs = {"prog_name", "hello there", "world"};

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs); 
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            {},
            {
              {key_posarg1, "hello there"},
              {key_posarg2, "world"},
            }
          };

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to be empty") {
          std::vector<std::string> inputs = {"prog_name", "hello", ""};

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs); 
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            {},
            {
              {key_posarg1, "hello"},
              {key_posarg2, ""},
            }
          };

          CHECK(result == correct);
        }
      }

      SUBCASE("with choices") {
        SUBCASE("choices is non-empty") {
          CLISpec cli = CLISpec{
            {},
            {
              CLIPositionalArgumentSpec{
                "posarg",
                std::vector<std::string>{"red", "blue", "green"},
                std::nullopt,
              },
            },
          };

          CLIPositionalArgumentKey key_posarg = CLIPositionalArgumentKey{0};

          SUBCASE("succeeds if a positional argument is set to a valid choice") {
            std::vector<std::string> inputs = {"prog_name", "blue"};

            tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              {},
              {
                {key_posarg, "red"},
              },
            };
          }

          SUBCASE("fails if a positional argument is set to an invalid choice") {
            std::vector<std::string> inputs = {"prog_name", " red"};

            tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = tl::unexpected("Invalid option for positional argument \"posarg\": \" red\"");

            CHECK(result == correct);
          }
        }

        SUBCASE("if choices is empty, rejects everything") {
          CLISpec cli = CLISpec{
            {},
            {
              CLIPositionalArgumentSpec{
                "posarg",
                std::vector<std::string>{},
                std::nullopt,
              },
            },
          };

          std::vector<std::string> inputs = {"prog_name", ""};

          tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected("Invalid option for positional argument \"posarg\": \"\"");

          CHECK(result == correct);
        }
      }
    }

    SUBCASE("correctly differentiates mixed arguments/flags") {
      CLISpec cli = CLISpec{
        {
          CLIFlagSpec{
            "flag1",
            'f',
            std::nullopt,
          },
          CLIFlagSpec{
            "flag2",
            std::nullopt,
            std::nullopt,
          },
          CLIFlagSpec{
            "flag3",
            'a',
            std::nullopt,
          },
        },
        {
          CLIPositionalArgumentSpec{
            "posarg1",
            std::vector<std::string>{"red", "blue", "green"},
            std::nullopt,
          },
          CLIPositionalArgumentSpec{
            "posarg2",
            std::nullopt,
            std::nullopt,
          },
        },
      };
      CLIFlagKey key_flag1 = CLIFlagKey{0};
      CLIFlagKey key_flag2 = CLIFlagKey{1};
      CLIFlagKey key_flag3 = CLIFlagKey{2};
      CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0};
      CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1};

      SUBCASE("works if flags are before positional arguments") {
        std::vector<std::string> inputs = {"prog_name", "-f", "--flag3", "red", "world"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, true},
            {key_flag2, false},
            {key_flag3, true},
          },
          {
            {key_posarg1, "red"},
            {key_posarg2, "world"},
          },
        };

        CHECK(result == correct);
      }

      SUBCASE("works if flags are interspersed") {
        std::vector<std::string> inputs = {"prog_name", "red", "-f", "world", "--flag3"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          {
            {key_flag1, true},
            {key_flag2, false},
            {key_flag3, true},
          },
          {
            {key_posarg1, "red"},
            {key_posarg2, "world"},
          },
        };

        CHECK(result == correct);
      }

      SUBCASE("detects if posargs are missing instead of treating flags as posarg values") {
        std::vector<std::string> inputs = {"prog_name", "-f", "red", "--flag2"};

        tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = tl::unexpected("Not enough positional arguments: found 1, expected 2");

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("cli_parse(CLISpec, int argc, char const * const *argv)") {
    // most cases are checked in the other overload,
    // i.e., cli_parse(CLISpec, std::vector<string>), 
    // so here we just throw in a single check to make sure
    // nothing has unexpectedly gone wrong
    CLISpec cli = CLISpec{
      {
        CLIFlagSpec{
          "flag1",
          'f',
          std::nullopt,
        },
        CLIFlagSpec{
          "flag2",
          std::nullopt,
          std::nullopt,
        },
        CLIFlagSpec{
          "flag3",
          'a',
          std::nullopt,
        },
      },
      {
        CLIPositionalArgumentSpec{
          "posarg1",
          std::vector<std::string>{"red", "blue", "green"},
          std::nullopt,
        },
        CLIPositionalArgumentSpec{
          "posarg2",
          std::nullopt,
          std::nullopt,
        },
      },
    };
    CLIFlagKey key_flag1 = CLIFlagKey{0};
    CLIFlagKey key_flag2 = CLIFlagKey{1};
    CLIFlagKey key_flag3 = CLIFlagKey{2};
    CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0};
    CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1};

    int argc = 5;
    char const *argv[] = {"prog_name", "red", "-f", "world", "--flag3"};

    tl::expected<CLIParseResult, std::string> result = cli_parse(cli, argc, argv);
    tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
      {
        {key_flag1, true},
        {key_flag2, false},
        {key_flag3, true},
      },
      {
        {key_posarg1, "red"},
        {key_posarg2, "world"},
      },
    };

    CHECK(result == correct);
  }
}
