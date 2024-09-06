#include "distributions.h"

namespace FlexFlow {

Constant::Constant(float val) : val(val) {}

float Constant::operator()() const {
  return val;
}

Uniform::Uniform(float a, float b) : a(a), b(b) {}

float Uniform::operator()() const {
  return a + ((static_cast<double>(std::rand()) / RAND_MAX) * (b - a));
}

Bernoulli::Bernoulli(float p) : p(p) {}

float Bernoulli::operator()() const {
  return (Uniform(0, 1)() < p);
}

Binary::Binary(float a, float b, float p) : a(a), b(b), p(p) {}

float Binary::operator()() const {
  return (Bernoulli(p)() ? a : b);
}

Chooser::Chooser(std::vector<float> items) : items(items) {}

float Chooser::operator()() const {
  return items[std::rand() % items.size()];
}

UniformNoise::UniformNoise(float lower, float upper)
    : lower(lower), upper(upper) {}

float UniformNoise::operator()() const {
  return Uniform(lower, upper)();
}

float NoNoise::operator()() const {
  return 1;
}

GaussianNoise::GaussianNoise(float mean, float stddev)
    : mean(mean), stddev(stddev) {}

float GaussianNoise::operator()() const {
  static std::default_random_engine generator;
  static std::normal_distribution<float> distribution(mean, stddev);
  return distribution(generator);
}

} // namespace FlexFlow
