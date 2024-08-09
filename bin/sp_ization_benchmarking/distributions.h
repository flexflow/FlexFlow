#ifndef _FLEXFLOW_DISTRIBUTIONS_H
#define _FLEXFLOW_DISTRIBUTIONS_H

#include "utils/graph/node/node.dtg.h"
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

struct Constant {
  float val;
  Constant(float val = 1);
  float operator()() const;
};

struct Uniform {
  float a, b;
  Uniform(float a = 0, float b = 1);
  float operator()() const;
};

struct Bernoulli {
  float p;
  Bernoulli(float p = 0.5);
  float operator()() const;
};

struct Binary {
  float a, b, p;
  Binary(float a = 0, float b = 1, float p = 0.5);
  float operator()() const;
};

struct UniformNoise {
  float lower, upper;
  UniformNoise(float lower = 0.9, float upper = 1.1);
  float operator()() const;
};

struct NoNoise {
  float operator()() const;
};

struct GaussianNoise {
  float mean, stddev;
  GaussianNoise(float mean = 1, float stddev = .1);
  float operator()() const;
};

template <typename Dist, typename Noise>
std::unordered_map<Node, float>
    make_cost_map(std::unordered_set<Node> const &nodes,
                  Dist const &distribution,
                  Noise const &noise) {
  std::unordered_map<Node, float> cost_map;
  for (Node const &node : nodes) {
    cost_map[node] = distribution() * noise();
  }
  return cost_map;
}

} // namespace FlexFlow

#endif
