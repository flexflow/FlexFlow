#ifndef _FLEXFLOW_DISTRIBUTIONS_H
#define _FLEXFLOW_DISTRIBUTIONS_H

#include "utils/graph/node/graph.h"
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

struct Constant {
  float val;
  Constant(float val = 1) : val(val) {}
  float operator()() const {
    return val;
  }
};

struct Uniform {
  float a, b;
  Uniform(float a = 0, float b = 1) : a(a), b(b) {}
  float operator()() const {
    return a + ((static_cast<double>(std::rand()) / RAND_MAX) * (b - a));
  }
};

struct Bernoulli {
  float p;
  Bernoulli(float p = 0.5) : p(p) {}
  float operator()() const {
    return (Uniform(0, 1)() < p);
  }
};

struct Binary {
  float a, b, p;
  Binary(float a = 0, float b = 1, float p = 0.5) : a(a), b(b), p(p) {}
  float operator()() const {
    return (Bernoulli(p)() ? a : b);
  }
};

struct UniformNoise {
  float lower, upper;
  UniformNoise(float lower = 0.9, float upper = 1.1)
      : lower(lower), upper(upper) {}
  float operator()() const {
    return Uniform(lower, upper)();
  }
};

struct NoNoise {
  float operator()() const {
    return 1;
  }
};

struct GaussianNoise {
  float mean, stddev;
  GaussianNoise(float mean = 1, float stddev = .1)
      : mean(mean), stddev(stddev) {}
  float operator()() const {
    static std::default_random_engine generator;
    static std::normal_distribution<float> distribution(mean, stddev);
    return distribution(generator);
  }
};

template <typename Dist, typename Noise = NoNoise>
std::unordered_map<Node, float>
    make_cost_map(std::unordered_set<Node> const &nodes,
                  Dist const &distribution,
                  Noise const &noise = NoNoise()) {
  std::unordered_map<Node, float> cost_map;
  for (Node const &node : nodes) {
    cost_map[node] = distribution() * noise();
  }
  return cost_map;
}

} // namespace FlexFlow

#endif
