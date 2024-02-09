#ifndef _FLEXFLOW_GRAPH_PARAMS_H_
#define _FLEXFLOW_GRAPH_PARAMS_H_

#include <stdio.h>
#include <string>

namespace FlexFlow {
    struct GraphParams {
    int num_active_requests;
    int num_active_tokens;
    bool generation_tokens;
    std::string graph_name; 

    GraphParams(int num_active_requests, int num_active_tokens, bool generation_tokens, const std::string & graph_name)
        : num_active_requests(num_active_requests), num_active_tokens(num_active_tokens), generation_tokens(generation_tokens),graph_name(graph_name) {}

    void Print() const {
        printf("GraphParams, num_active_requests: %d, num_active_tokens: %d, generation_tokens: %d\n graph_name:%s\n", num_active_requests, num_active_tokens, generation_tokens, graph_name.c_str());
    }
    };



}

namespace std {
  template <>
  struct hash<FlexFlow::GraphParams> {
    size_t operator()(const FlexFlow::GraphParams& gp) const {
      return std::hash<int>()(gp.num_active_requests) ^
             std::hash<int>()(gp.num_active_tokens) ^
             std::hash<bool>()(gp.generation_tokens)^
             std::hash<std::string>()(gp.graph_name) ;
    }
  };
}

namespace std {
  template <>
  struct equal_to<FlexFlow::GraphParams> {
    bool operator()(const FlexFlow::GraphParams& lhs, const FlexFlow::GraphParams& rhs) const {
      return lhs.num_active_requests == rhs.num_active_requests &&
             lhs.num_active_tokens == rhs.num_active_tokens && 
             lhs.generation_tokens == rhs.generation_tokens &&
                lhs.graph_name == rhs.graph_name;
    }
  };
}

#endif