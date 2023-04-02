#include "runtime/config.h"
#include "legion.h"

using namespace Legion;

namespace FlexFlow {

FFConfig::FFConfig() {
  // Use Real::Machine::get_address_space_count() to obtain the number of nodes
  numNodes = Realm::Machine::get_machine().get_address_space_count();

  Runtime *runtime = Runtime::get_runtime();
  lg_hlr = runtime;
  lg_ctx = Runtime::get_context();
  field_space = runtime->create_field_space(lg_ctx);
}


}
