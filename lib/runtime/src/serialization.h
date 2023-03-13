#ifndef _FLEXFLOW_RUNTIME_SERIALIZATION_H
#define _FLEXFLOW_RUNTIME_SERIALIZATION_H

#include "legion.h"
#include "compiler/compiler.h"

namespace FlexFlow {
  void serialize(Legion::Serializer const &, SearchSolution const &);
  void deserialize(Legion::Deserializer const &, SearchSolution &);

  /* void deserialize_graph_optimal_view( */
  /*     Legion::Deserializer &dez, */
  /*     PCG::Graph *graph, */
  /*     std::unordered_map<PCG::Node, MachineView> &optimal_views); */
}

#endif
