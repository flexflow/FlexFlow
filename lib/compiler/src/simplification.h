#ifndef _FLEXFLOW_FFC_SIMPLIFICATION_H
#define _FLEXFLOW_FFC_SIMPLIFICATION_H

#include "graph.h"
#include "spdlog/spdlog.h"
#include <string>

namespace FlexFlow {
namespace PCG {

struct SimplificationSettings {
  bool simplify_parallel_ops = false;
  bool fuse_parallel_ops = false;
  bool remove_trailing_parallel_ops = false;
  bool remove_noops = false;
};

class Simplifier {
public:
  Simplifier(std::string const &logger_name);

  Graph const &simplify(SimplificationSettings const &, Graph const &);

private:
  void simplify_parallel_ops();

private:
  std::shared_ptr<spdlog::logger> logger;
};

} // namespace PCG
} // namespace FlexFlow

#endif
