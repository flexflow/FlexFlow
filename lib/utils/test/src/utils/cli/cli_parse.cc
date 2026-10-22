#include "utils/cli/cli_parse.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cli_parse(CLISpec, std::vector<std::string>)") {
    SUBCASE("works even if cli is empty") {
      CLISpec cli = CLISpec{/*flags=*/{},
                            /*named_arguments=*/{},
                            /*positional_arguments=*/{}};
      std::vector<std::string> inputs = {"prog_name"};

      tl::expected<CLIParseResult, std::string> result = cli_parse(cli, inputs);
      tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
          /*flags=*/{},
          /*named_arguments=*/{},
          /*positional_arguments=*/{},
      };

      CHECK(result == correct);
    }

    SUBCASE("flag parsing") {
      CLISpec cli = CLISpec{
          /*flags=*/{
              CLIFlagSpec{
                  /*long_flag=*/"flag1",
                  /*short_flag=*/std::nullopt,
                  /*description=*/std::nullopt,
              },
              CLIFlagSpec{
                  /*long_flag=*/"flag2",
                  /*short_flag=*/'2',
                  /*description=*/std::nullopt,
              },
          },
          /*named_arguments=*/{},
          /*positional_arguments=*/{},
      };
      CLIFlagKey key_flag1 = CLIFlagKey{0_n};
      CLIFlagKey key_flag2 = CLIFlagKey{1_n};

      SUBCASE("parses flags in any order") {
        std::vector<std::string> inputs = {"prog_name", "-2", "--flag1"};

        tl::expected<CLIParseResult, std::string> result =
            cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            /*flags=*/{
                {key_flag1, true},
                {key_flag2, true},
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        CHECK(result == correct);
      }

      SUBCASE("does not allow single-dash long flags") {
        std::vector<std::string> inputs = {"prog_name", "-flag1"};

        tl::expected<CLIParseResult, std::string> result =
            cli_parse(cli, inputs);

        tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
            "Encountered unknown flag or named argument: -flag1");

        CHECK(result == correct);
      }

      SUBCASE("is fine if some are not present") {
        std::vector<std::string> inputs = {"prog_name", "-2"};

        tl::expected<CLIParseResult, std::string> result =
            cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            /*flags=*/{
                {key_flag1, false},
                {key_flag2, true},
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        CHECK(result == correct);
      }

      SUBCASE("is fine if none are present") {
        std::vector<std::string> inputs = {"prog_name"};

        tl::expected<CLIParseResult, std::string> result =
            cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            /*flags=*/{
                {key_flag1, false},
                {key_flag2, false},
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        CHECK(result == correct);
      }

      SUBCASE("is fine even if the program name is a flag") {
        std::vector<std::string> inputs = {"--flag1", "-2"};

        tl::expected<CLIParseResult, std::string> result =
            cli_parse(cli, inputs);
        tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
            /*flags=*/{
                {key_flag1, false},
                {key_flag2, true},
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("named argument parsing") {
      SUBCASE("without choices") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg1",
                    /*metavar=*/"NAMEDARG1",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg2",
                    /*metavar=*/"NAMEDARG2",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/{},
        };

        CLINamedArgumentKey key_namedarg1 = CLINamedArgumentKey{0_n};
        CLINamedArgumentKey key_namedarg2 = CLINamedArgumentKey{1_n};

        SUBCASE("can parse multiple named arguments") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg1", "hello", "--namedarg2", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, "hello"},
                  {key_namedarg2, "world"},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("parses flags in any order") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg2", "world", "--namedarg1", "hello"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, "hello"},
                  {key_namedarg2, "world"},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("is fine if some are not present") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg2", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, std::nullopt},
                  {key_namedarg2, "world"},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("prints correct error message if value is missing") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg1", "--namedarg2", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Missing value for named argument \"--namedarg1\"");

          CHECK(result == correct);
        }

        SUBCASE("prints correct error message if value is missing and is the "
                "last key") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg1", "hello", "--namedarg2"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Missing value for named argument \"--namedarg2\"");

          CHECK(result == correct);
        }

        SUBCASE("is fine if none are present") {
          std::vector<std::string> inputs = {"prog_name"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, std::nullopt},
                  {key_namedarg2, std::nullopt},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("is fine even if the program name is a flag") {
          std::vector<std::string> inputs = {
              "--namedarg1", "--namedarg2", "hello"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, std::nullopt},
                  {key_namedarg2, "hello"},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to contain spaces") {
          std::vector<std::string> inputs = {"prog_name",
                                             "--namedarg1",
                                             "hello there",
                                             "--namedarg2",
                                             "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, "hello there"},
                  {key_namedarg2, "world"},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to be empty") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg1", "hello", "--namedarg2", ""};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  {key_namedarg1, "hello"},
                  {key_namedarg2, ""},
              },
              /*positional_arguments=*/{},
          };

          CHECK(result == correct);
        }
      }

      SUBCASE("with choices") {
        SUBCASE("choices is non-empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  CLINamedArgumentSpec{
                      /*long_flag=*/"namedarg",
                      /*metavar=*/"NAMEDARG",
                      /*choices=*/
                      std::vector<std::string>{"red", "blue", "green"},
                      /*description=*/std::nullopt,
                  },
              },
              /*positional_arguments=*/{},
          };

          CLINamedArgumentKey key_namedarg = CLINamedArgumentKey{0_n};

          SUBCASE("succeeds if a named argument is set to a valid choice") {
            std::vector<std::string> inputs = {
                "prog_name", "--namedarg", "blue"};

            tl::expected<CLIParseResult, std::string> result =
                cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
                /*flags=*/{},
                /*named_arguments=*/
                {
                    {key_namedarg, "blue"},
                },
                /*positional_arguments=*/{},
            };
          }

          SUBCASE("fails if a named argument argument is set to an invalid "
                  "choice") {
            std::vector<std::string> inputs = {
                "prog_name", "--namedarg", " red"};

            tl::expected<CLIParseResult, std::string> result =
                cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
                "Invalid option for named argument \"--namedarg\": \" red\"");

            CHECK(result == correct);
          }
        }

        SUBCASE("if choices is empty, rejects everything") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  CLINamedArgumentSpec{
                      /*long_flag=*/"namedarg",
                      /*metavar=*/"N",
                      /*choices=*/std::vector<std::string>{},
                      /*description=*/std::nullopt,
                  },
              },
              /*positional_arguments=*/{},
          };

          std::vector<std::string> inputs = {"prog_name", "--namedarg", ""};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Invalid option for named argument \"--namedarg\": \"\"");

          CHECK(result == correct);
        }
      }
    }

    SUBCASE("positional argument parsing") {
      SUBCASE("without choices") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/{},
            /*positional_arguments=*/
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

        CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0_n};
        CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1_n};

        SUBCASE("can parse multiple positional arguments") {
          std::vector<std::string> inputs = {"prog_name", "hello", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  {key_posarg1, "hello"},
                  {key_posarg2, "world"},
              },
          };

          CHECK(result == correct);
        }

        SUBCASE("requires all positional arguments to be present") {
          std::vector<std::string> inputs = {"prog_name", "hello"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Not enough positional arguments: found 1, expected 2");

          CHECK(result == correct);
        }

        SUBCASE("requires no extra positional arguments to be present") {
          std::vector<std::string> inputs = {
              "prog_name", "hello", "there", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct =
              tl::unexpected("Too many positional arguments: expected 2");

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to contain spaces") {
          std::vector<std::string> inputs = {
              "prog_name", "hello there", "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  {key_posarg1, "hello there"},
                  {key_posarg2, "world"},
              },
          };

          CHECK(result == correct);
        }

        SUBCASE("allows arguments to be empty") {
          std::vector<std::string> inputs = {"prog_name", "hello", ""};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  {key_posarg1, "hello"},
                  {key_posarg2, ""},
              },
          };

          CHECK(result == correct);
        }
      }

      SUBCASE("with choices") {
        SUBCASE("choices is non-empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  CLIPositionalArgumentSpec{
                      "posarg",
                      std::vector<std::string>{"red", "blue", "green"},
                      std::nullopt,
                  },
              },
          };

          CLIPositionalArgumentKey key_posarg = CLIPositionalArgumentKey{0_n};

          SUBCASE(
              "succeeds if a positional argument is set to a valid choice") {
            std::vector<std::string> inputs = {"prog_name", "blue"};

            tl::expected<CLIParseResult, std::string> result =
                cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    {key_posarg, "red"},
                },
            };
          }

          SUBCASE(
              "fails if a positional argument is set to an invalid choice") {
            std::vector<std::string> inputs = {"prog_name", " red"};

            tl::expected<CLIParseResult, std::string> result =
                cli_parse(cli, inputs);
            tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
                "Invalid option for positional argument \"posarg\": \" red\"");

            CHECK(result == correct);
          }
        }

        SUBCASE("if choices is empty, rejects everything") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  CLIPositionalArgumentSpec{
                      "posarg",
                      std::vector<std::string>{},
                      std::nullopt,
                  },
              },
          };

          std::vector<std::string> inputs = {"prog_name", ""};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Invalid option for positional argument \"posarg\": \"\"");

          CHECK(result == correct);
        }
      }
    }

    SUBCASE("mixed arguments/flags") {
      SUBCASE("correctly differentiates mixed arguments/flags") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/'f',
                    /*description=*/std::nullopt,
                },
                CLIFlagSpec{
                    /*long_flag=*/"flag2",
                    /*short_flag=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
                CLIFlagSpec{
                    /*long_flag=*/"flag3",
                    /*short_flag=*/'a',
                    /*description=*/std::nullopt,
                },
            },
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg1",
                    /*metavar=*/"NAMEDARG1",
                    /*choices=*/std::vector<std::string>{"one", "two", "three"},
                    /*description=*/std::nullopt,
                },
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg2",
                    /*metavar=*/"NAMEDARG2",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg1",
                    /*choices=*/
                    std::vector<std::string>{"red", "blue", "green"},
                    /*description=*/std::nullopt,
                },
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg2",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
        };

        CLIFlagKey key_flag1 = CLIFlagKey{0_n};
        CLIFlagKey key_flag2 = CLIFlagKey{1_n};
        CLIFlagKey key_flag3 = CLIFlagKey{2_n};
        CLINamedArgumentKey key_namedarg1 = CLINamedArgumentKey{0_n};
        CLINamedArgumentKey key_namedarg2 = CLINamedArgumentKey{1_n};
        CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0_n};
        CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1_n};

        SUBCASE("works if flags and named arguments are before positional "
                "arguments") {
          std::vector<std::string> inputs = {"prog_name",
                                             "-f",
                                             "--flag3",
                                             "--namedarg1",
                                             "two",
                                             "red",
                                             "world"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{
                  {key_flag1, true},
                  {key_flag2, false},
                  {key_flag3, true},
              },
              /*named_arguments=*/
              {
                  {key_namedarg1, "two"},
                  {key_namedarg2, std::nullopt},
              },
              /*positional_arguments=*/
              {
                  {key_posarg1, "red"},
                  {key_posarg2, "world"},
              },
          };

          CHECK(result == correct);
        }

        SUBCASE("works if flags and named arguments are interspersed") {
          std::vector<std::string> inputs = {"prog_name",
                                             "red",
                                             "-f",
                                             "--namedarg1",
                                             "two",
                                             "world",
                                             "--flag3"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
              /*flags=*/{
                  {key_flag1, true},
                  {key_flag2, false},
                  {key_flag3, true},
              },
              /*named_arguments=*/
              {
                  {key_namedarg1, "two"},
                  {key_namedarg2, std::nullopt},
              },
              /*positional_arguments=*/
              {
                  {key_posarg1, "red"},
                  {key_posarg2, "world"},
              },
          };

          CHECK(result == correct);
        }

        SUBCASE("detects if posargs are missing instead of treating flags as "
                "posarg values") {
          std::vector<std::string> inputs = {
              "prog_name", "-f", "red", "--flag2"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Not enough positional arguments: found 1, expected 2");

          CHECK(result == correct);
        }

        SUBCASE("detects if posargs are missing instead of treating named "
                "arguments as "
                "posarg values") {
          std::vector<std::string> inputs = {
              "prog_name", "--namedarg1", "two", "red", "--flag2"};

          tl::expected<CLIParseResult, std::string> result =
              cli_parse(cli, inputs);
          tl::expected<CLIParseResult, std::string> correct = tl::unexpected(
              "Not enough positional arguments: found 1, expected 2");

          CHECK(result == correct);
        }
      }
    }
  }

  TEST_CASE("cli_parse(CLISpec, int argc, char const star const star argv)") {
    // most cases are checked in the other overload,
    // i.e., cli_parse(CLISpec, std::vector<string>),
    // so here we just throw in a single check to make sure
    // nothing has unexpectedly gone wrong
    CLISpec cli = CLISpec{
        /*flags=*/{
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
        /*named_arguments=*/
        {
            CLINamedArgumentSpec{
                /*long_flag=*/"namedarg1",
                /*metavar=*/"NAMEDARG1",
                /*choices=*/std::vector<std::string>{"one", "two", "three"},
                /*description=*/std::nullopt,
            },
            CLINamedArgumentSpec{
                /*long_flag=*/"namedarg2",
                /*metavar=*/"NAMEDARG2",
                /*choices=*/std::nullopt,
                /*description=*/std::nullopt,
            },
        },
        /*positional_arguments=*/
        {
            CLIPositionalArgumentSpec{
                /*name=*/"posarg1",
                /*choices=*/std::vector<std::string>{"red", "blue", "green"},
                /*description=*/std::nullopt,
            },
            CLIPositionalArgumentSpec{
                /*name=*/"posarg2",
                /*choices=*/std::nullopt,
                /*description=*/std::nullopt,
            },
        },
    };

    CLIFlagKey key_flag1 = CLIFlagKey{0_n};
    CLIFlagKey key_flag2 = CLIFlagKey{1_n};
    CLIFlagKey key_flag3 = CLIFlagKey{2_n};
    CLINamedArgumentKey key_namedarg1 = CLINamedArgumentKey{0_n};
    CLINamedArgumentKey key_namedarg2 = CLINamedArgumentKey{1_n};
    CLIPositionalArgumentKey key_posarg1 = CLIPositionalArgumentKey{0_n};
    CLIPositionalArgumentKey key_posarg2 = CLIPositionalArgumentKey{1_n};

    int argc = 7;
    char const *argv[] = {
        "prog_name", "red", "-f", "--namedarg1", "two", "world", "--flag3"};

    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    tl::expected<CLIParseResult, std::string> correct = CLIParseResult{
        /*flags=*/{
            {key_flag1, true},
            {key_flag2, false},
            {key_flag3, true},
        },
        /*named_arguments=*/
        {
            {key_namedarg1, "two"},
            {key_namedarg2, std::nullopt},
        },
        /*positional_arguments=*/
        {
            {key_posarg1, "red"},
            {key_posarg2, "world"},
        },
    };

    CHECK(result == correct);
  }
}
