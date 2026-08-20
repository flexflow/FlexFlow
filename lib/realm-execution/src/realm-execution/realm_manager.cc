#include "realm-execution/realm_manager.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/redops/realm_redop_registry.h"
#include "realm-execution/tasks/realm_task_registry.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv)
    : RealmContext(Realm::Processor::NO_PROC) {
  bool ok = this->get_runtime().init(argc, argv);
  ASSERT(ok);

  // Register all tasks and redops at initialization time so we don't need to later
  register_all_tasks().wait();
  register_all_redops();
}

RealmManager::~RealmManager() {
  Realm::Event outstanding = this->merge_outstanding_events();
  this->get_runtime().shutdown(outstanding);
  this->get_runtime().wait_for_shutdown();
}

ControllerTaskResult
    RealmManager::start_controller(std::function<void(RealmContext &)> thunk,
                                   Realm::Event wait_on) {

  Realm::Processor target_proc =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::LOC_PROC)
          .first();

  return collective_spawn_controller_task(*this, target_proc, thunk, wait_on);
}

} // namespace FlexFlow
