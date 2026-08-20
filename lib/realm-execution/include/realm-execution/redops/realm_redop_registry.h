#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REDOPS_REALM_REDOP_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REDOPS_REALM_REDOP_REGISTRY_H

namespace FlexFlow {

/**
 * \brief Registers all known reduction operators (redops), fetching the
 * process-global Realm runtime itself.
 *
 * \note Deliberately takes no PRealm-typed argument and is implemented in a
 * .cu file (needed to compile GPU reduction kernels for the redops) —
 * including realm-execution/realm.h (i.e. PRealm's prealm.h) from a .cu
 * translation unit fails to compile under nvcc.
 */
void register_all_redops();

} // namespace FlexFlow

#endif
