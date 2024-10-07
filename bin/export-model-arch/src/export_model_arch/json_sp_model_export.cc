#include "export_model_arch/json_sp_model_export.h"

using namespace ::FlexFlow;

namespace nlohmann {

JsonSPModelExport adl_serializer<JsonSPModelExport>::from_json(json const &j) {
  NOT_IMPLEMENTED(); 
}

static void sp_decomposition_to_json(json &j, LeafOnlyBinarySPDecompositionTree<int> const &t) {
}

void adl_serializer<JsonSPModelExport>::to_json(json &j, JsonSPModelExport const &m) {
  j["computation_graph"] = m.computation_graph;
  sp_decomposition_to_json(j["sp_decomposition"], m.sp_decomposition);
}


} // namespace nlohmann
