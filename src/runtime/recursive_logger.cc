#include "flexflow/utils/recursive_logger.h"

namespace FlexFlow {

RecursiveLogger::RecursiveLogger(std::string const &category_name)
    : logger(category_name) {}

Realm::LoggerMessage RecursiveLogger::info() {
  Realm::LoggerMessage msg = this->logger.info();
  this->print_prefix(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::debug() {
  Realm::LoggerMessage msg = this->logger.debug();
  this->print_prefix(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::spew() {
  Realm::LoggerMessage msg = this->logger.spew();
  this->print_prefix(msg);
  return msg;
}

void RecursiveLogger::print_prefix(Realm::LoggerMessage &msg) const {
  msg << this->depth << " ";
  for (int i = 0; i < this->depth; i++) {
    msg << " ";
  }
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

}; // namespace FlexFlow
