#ifndef _FLEXFLOW_RUNTIME_SRC_LOGGERS_H
#define _FLEXFLOW_RUNTIME_SRC_LOGGERS_H

#include "legion/legion_utilities.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_profile, log_measure, log_sim,
    log_ps_sim, log_xfer_sim, log_xfer_est, log_metrics, log_model, log_mapper;

}

#endif
