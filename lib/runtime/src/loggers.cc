#include "loggers.h"

namespace FlexFlow {

LegionRuntime::Logger::Category log_profile("profile"), log_measure("measure"),
    log_sim("sim"), log_ps_sim("ps_sim"), log_xfer_sim("xfer_sim"),
    log_xfer_est("xfer_est"), log_metrics("metrics"), log_model("Model"),
    log_mapper("Mapper");

}
