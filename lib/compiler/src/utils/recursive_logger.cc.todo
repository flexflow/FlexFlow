#include "utils/recursive_logger.h"
#include "utils/exception.h"

namespace FlexFlow {
namespace utils {

RecursiveLogger::RecursiveLogger(std::shared_ptr<spdlog::logger> const &logger)
    : logger(logger) {}

RecursiveLogger::RecursiveLogger(std::string const &logger_name) {
  this->logger = spdlog::get(logger_name);
}

std::string RecursiveLogger::get_prefix() const {
  return std::string(this->depth * 2, ' ');
}

void RecursiveLogger::enter() {
  this->depth++;
}

void RecursiveLogger::leave() {
  this->depth--;
  assert(this->depth >= 0);
}

std::unique_ptr<DepthTag> RecursiveLogger::enter_tag() {
  return std::unique_ptr<DepthTag>(new DepthTag(*this));
}

DepthTag::DepthTag(RecursiveLogger &_logger) : logger(_logger) {
  this->logger.enter();
}

DepthTag::~DepthTag() {
  this->logger.leave();
}

} // namespace utils
} // namespace FlexFlow
