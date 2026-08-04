#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_NCCL_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_NCCL_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include <string>
#include <cstddef>

namespace FlexFlow {
    void nccl_task_body(void const *args,
        size_t arglen,
        void const *userdata,
        size_t userdata_len,
        Realm::Processor proc);


Realm::Event spawn_nccl_task(RealmContext &ctx,
    Realm::Processor target_proc,
    std::string const &message,
    Realm::Event precondition);

}

#endif
