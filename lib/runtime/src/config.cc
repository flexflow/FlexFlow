#include "runtime/config.h"
#include "legion.h"

using namespace Legion;

namespace FlexFlow {

LegionConfig::LegionConfig() 
  : lg_ctx(Runtime::get_context()), lg_hlr(Runtime::get_runtime())
{
  this->field_space = this->lg_hlr->create_field_space(this->lg_ctx); 
}

FFConfig::FFConfig() {
  // Use Realm::Machine::get_address_space_count() to obtain the number of nodes
  numNodes = Realm::Machine::get_machine().get_address_space_count();
}


}
