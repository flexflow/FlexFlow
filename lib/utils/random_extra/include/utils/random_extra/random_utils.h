#ifndef _RANDOM_UTILS_H
#define _RANDOM_UTILS_H

#include <cstdlib>
#include <stdexcept>
#include <vector>

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

template <typename T>
T select_random(std::vector<T> const &values) {
  return values[std::rand() % values.size()];
}

template <typename T>
T select_random_determistic(std::vector<T> const &values,
                            std::vector<float> const &weights,
                            float value) {
  if (values.empty()) {
    throw std::invalid_argument("Values list must not be empty.");
  }
  float total = 0.0f;
  for (auto const &w : weights) {
    if (w < 0) {
      throw std::invalid_argument("Weights must not be negative");
    }
    total += w;
  }

  float r = value * total;
  float curr = 0.0f;
  int i = -1;
  while (curr <= r && (i < 0 || i < (int)values.size() - 1)) {
    i++;
    curr += weights[i];
  }
  return values[i];
}

template <typename T>
T select_random(std::vector<T> const &values,
                std::vector<float> const &weights) {
  return select_random_determistic<T>(values, weights, randf());
}

#endif // _RANDOM_UTILS_H
