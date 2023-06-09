#ifndef _FLEXFLOW_RECURSIVE_LOGGER_H
#define _FLEXFLOW_RECURSIVE_LOGGER_H

#include "spdlog/spdlog.h"
#include <memory>

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define UNIQUE_TAG() CONCAT(tag, __COUNTER__)
#define TAG_ENTER(mlogger) auto UNIQUE_TAG() = mlogger->enter_tag()

namespace FlexFlow {
namespace utils {

class RecursiveLogger;

class DepthTag {
public:
  DepthTag() = delete;
  DepthTag(RecursiveLogger &);
  DepthTag(DepthTag const &) = delete;
  ~DepthTag();

private:
  RecursiveLogger &logger;
};

class RecursiveLogger {
public:
  RecursiveLogger(std::shared_ptr<spdlog::logger> const &logger);
  RecursiveLogger(std::string const &logger_name);

  RecursiveLogger(RecursiveLogger const &) = delete;

  template <typename... Args>
  void info(std::string const &fmt, Args &&...args) {
    this->logger->info(this->get_prefix() + fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void debug(std::string const &fmt, Args &&...args) {
    this->logger->debug(this->get_prefix() + fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void spew(std::string const &fmt, Args &&...args) {
    this->logger->trace(this->get_prefix() + fmt, std::forward<Args>(args)...);
  }

  void enter();
  void leave();

  std::unique_ptr<DepthTag> enter_tag();

private:
  std::string get_prefix() const;

private:
  int depth = 0;
  std::shared_ptr<spdlog::logger> logger;
};

} // namespace utils
} // namespace FlexFlow

#endif
