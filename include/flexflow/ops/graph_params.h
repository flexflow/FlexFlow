#ifndef _FLEXFLOW_GRAPH_PARAMS_H_
#define _FLEXFLOW_GRAPH_PARAMS_H_

#include <stdio.h>
#include <string>

namespace FlexFlow {
  struct GraphParams {
    int num_active_requests;
    int num_active_tokens;
    bool prompt_phase;

    GraphParams(int num_active_requests, int num_active_tokens, bool prompt_phase)
      : num_active_requests(num_active_requests), num_active_tokens(num_active_tokens), prompt_phase(prompt_phase) {}

    void Print() const {
      printf("GraphParams, num_active_requests: %d, num_active_tokens: %d, prompt_phase: %d\n\n", num_active_requests, num_active_tokens, prompt_phase);
    }
  };

}

namespace std {
  template <>
  struct hash<FlexFlow::GraphParams> {
    size_t operator()(const FlexFlow::GraphParams& gp) const {
      return std::hash<int>()(gp.num_active_requests) ^
             std::hash<int>()(gp.num_active_tokens) ^
             std::hash<bool>()(gp.prompt_phase);
    }
  };
}

namespace std {
  template <>
  struct equal_to<FlexFlow::GraphParams> {
    bool operator()(const FlexFlow::GraphParams& lhs, const FlexFlow::GraphParams& rhs) const {
      return lhs.num_active_requests == rhs.num_active_requests &&
             lhs.num_active_tokens == rhs.num_active_tokens && 
             lhs.prompt_phase == rhs.prompt_phase;
    }
  };
}

#endif
