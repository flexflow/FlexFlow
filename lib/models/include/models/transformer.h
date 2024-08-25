#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_H

#include "models/transformer.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

class Transformer {
public:
  Transformer(TransformerConfig const &config) : config_(config) {
    init_model();
  }

  void init_model();

  [[nodiscard]] ComputationGraph get_computation_graph() const;

private:
private:
  TransformerConfig config_;
  ComputationGraphBuilder cgb_;
};

ComputationGraph get_transformer_computation_graph(TransformerConfig const &);

} // namespace FlexFlow

#endif
