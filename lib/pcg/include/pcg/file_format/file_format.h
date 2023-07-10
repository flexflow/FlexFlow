#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_FILE_FORMAT_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_FILE_FORMAT_H

#include "graphs.h"
#include "utils/json.h"

namespace FlexFlow {

enum class FileFormatVersion {
  V1,
  UNSTABLE,
};

json to_json(ComputationGraph const &, FileFormatVersion);
ComputationGraph from_json(json const &);

} // namespace FlexFlow

#endif
