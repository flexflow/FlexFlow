#ifndef _FLEXFLOW_BIN_EXPORT_MODEL_ARCH_INCLUDE_EXPORT_MODEL_ARCH_JSON_SP_MODEL_EXPORT_H
#define _FLEXFLOW_BIN_EXPORT_MODEL_ARCH_INCLUDE_EXPORT_MODEL_ARCH_JSON_SP_MODEL_EXPORT_H

#include <nlohmann/json.hpp>
#include "export_model_arch/json_sp_model_export.dtg.h"

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::JsonSPModelExport> {
  static ::FlexFlow::JsonSPModelExport from_json(json const &);
  static void to_json(json &, ::FlexFlow::JsonSPModelExport const &);
};

} // namespace nlohmann

#endif
