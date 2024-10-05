#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_resource_splits") {
    auto make_machine_spec = [](int num_nodes, int num_gpus_per_node) {
      return MachineSpecification{
          /*num_nodes=*/num_nodes,
          /*num_cpus_per_node=*/1,
          /*num_gpus_per_node=*/num_gpus_per_node,
          /*inter_node_bandwidth=*/1.0,
          /*intra_node_bandwidth=*/1.0,
      };
    };

    SUBCASE("returns no splits if no splits are possible") {
      MachineSpecification input = make_machine_spec(/*num_nodes=*/1,
                                                     /*num_gpus_per_node=*/1);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          result = get_machine_resource_splits(input);
      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          correct = {};

      CHECK(result == correct);
    }

    SUBCASE(
        "returns splits in gpu and node dimensions, but not at the same time") {
      MachineSpecification input = make_machine_spec(/*num_nodes=*/2,
                                                     /*num_gpus_per_node=*/2);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          result = get_machine_resource_splits(input);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          correct = {
              {
                  make_machine_spec(/*num_nodes=*/2,
                                    /*num_gpus_per_node=*/1),
                  make_machine_spec(/*num_nodes=*/2,
                                    /*num_gpus_per_node=*/1),
              },
              {
                  make_machine_spec(/*num_nodes=*/1,
                                    /*num_gpus_per_node=*/2),
                  make_machine_spec(/*num_nodes=*/1,
                                    /*num_gpus_per_node=*/2),
              },

          };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in node dimension in powers of two") {
      SUBCASE("num_nodes is a power of 2") {
        MachineSpecification input = make_machine_spec(/*num_nodes=*/8,
                                                       /*num_gpus_per_node=*/1);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/7,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/2,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/6,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/4,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/4,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/6,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/2,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/7,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_nodes is not a power of 2") {
        MachineSpecification input = make_machine_spec(/*num_nodes=*/6,
                                                       /*num_gpus_per_node=*/1);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/5,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/2,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/4,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/4,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/2,
                                      /*num_gpus_per_node=*/1),
                },
                {
                    make_machine_spec(/*num_nodes=*/5,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                },
            };

        CHECK(result == correct);
      }
    }

    SUBCASE("returns splits in gpu dimension in powers of two") {
      SUBCASE("num_gpus_per_node is a power of 2") {
        MachineSpecification input = make_machine_spec(/*num_nodes=*/1,
                                                       /*num_gpus_per_node=*/8);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/7),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/2),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/6),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/4),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/4),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/6),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/2),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/7),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_gpus_per_node is not a power of 2") {
        MachineSpecification input = make_machine_spec(/*num_nodes=*/1,
                                                       /*num_gpus_per_node=*/6);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/5),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/2),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/4),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/4),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/2),
                },
                {
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/5),
                    make_machine_spec(/*num_nodes=*/1,
                                      /*num_gpus_per_node=*/1),
                },
            };
      }
    }
  }
}
