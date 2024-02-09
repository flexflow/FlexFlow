#ifndef _FLEXFLOW_GRAPH_PARAMS_H_
#define _FLEXFLOW_GRAPH_PARAMS_H_

namespace FlexFlow {
    struct GraphParams {
    int num_active_requests;
    int num_active_tokens;
    bool generation_tokens;

    GraphParams(int num_active_requests, int num_active_tokens, bool generation_tokens)
        : num_active_requests(num_active_requests), num_active_tokens(num_active_tokens), generation_tokens(generation_tokens) {}
    };

}

namespace std {
  template <>
  struct hash<FlexFlow::GraphParams> {
    size_t operator()(const FlexFlow::GraphParams& gp) const {
      return std::hash<int>()(gp.num_active_requests) ^
             std::hash<int>()(gp.num_active_tokens) ^
             std::hash<bool>()(gp.generation_tokens);
    }
  };
}

namespace std {
  template <>
  struct equal_to<FlexFlow::GraphParams> {
    bool operator()(const FlexFlow::GraphParams& lhs, const FlexFlow::GraphParams& rhs) const {
      return lhs.num_active_requests == rhs.num_active_requests &&
             lhs.num_active_tokens == rhs.num_active_tokens &&
             lhs.generation_tokens == rhs.generation_tokens;
    }
  };
}

#endif