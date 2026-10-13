#include "internal/realm_test_utils.h"
#include "realm-execution/realm_manager.h"
#include "realm-execution/tasks/impl/nccl_task.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("NCCL task prints Hello World") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/1_n);

    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager(&fake_argc, &fake_argv);

    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Realm::Event event = spawn_nccl_task(
              ctx,
              ctx.get_current_processor(),
              "Hello World from NCCL!",
              Realm::Event::NO_EVENT);

          event.wait();
        });

    result.wait();
  }
}

} // namespace test
