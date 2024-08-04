#include "flexflow/operator.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/simulator.h"
#include <stdexcept>
#include <wordexp.h>
#include <unistd.h>

namespace FlexFlow {

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  throw std::runtime_error(
      "No overload of get_params_hash defined for op type " +
      get_operator_type_name(this->op_type));
}

fs::path get_dst_folder(std::string const &subdir,
                        int step_idx,
                        int shard_idx,
                        bool before_kernel) {
  std::vector<std::string> debug_subdirs = {"fwd", "bwd", "optim", "weights"};
  assert(std::find(debug_subdirs.begin(), debug_subdirs.end(), subdir) !=
         debug_subdirs.end());
  std::string step_substr = "step_" + std::to_string(step_idx);
  if (before_kernel) {
    step_substr += "_pre";
  }
  char cwd[PATH_MAX];
  getcwd(cwd, sizeof(cwd));

  // char const *ff_cache_path = std::string(std::getenv("FF_DEBUG_PATH")) == "." ?
  //     cwd : std::getenv("FF_DEBUG_PATH");

  char const *ff_cache_path = std::getenv("FF_CACHE_PATH");
  
  std::string debug_dir_ =
      ff_cache_path ? std::string(ff_cache_path) + "/debug/flexflow"
                    : std::string("~/.cache/flexflow/debug/flexflow");
  wordexp_t p;
  wordexp(debug_dir_.c_str(), &p, 0);
  debug_dir_ = p.we_wordv[0];
  wordfree(&p);
  fs::path debug_dir = debug_dir_;
  if(!fs::is_directory(debug_dir)) {
    printf("invalid debug directory: %s\n", debug_dir.c_str());
  }
  assert(fs::is_directory(debug_dir));
  fs::path dst_folder =
      debug_dir / subdir / step_substr / ("shard_" + std::to_string(shard_idx));
  fs::create_directories(dst_folder);
  return dst_folder;
}

}; // namespace FlexFlow