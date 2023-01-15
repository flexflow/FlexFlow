#include "utils/recursive_logger.h"
#include "utils/not_implemented_exception.h"

namespace FlexFlow {

RecursiveLogger::RecursiveLogger(std::string const &category_name) {}

std::ostream &RecursiveLogger::info() {
  throw NotImplemented();
  /* Realm::LoggerMessage msg = this->logger.info(); */
  /* this->print_prefix(msg); */
  /* return msg; */
}

std::ostream &RecursiveLogger::debug() {
  throw NotImplemented();
  /* Realm::LoggerMessage msg = this->logger.debug(); */
  /* this->print_prefix(msg); */
  /* return msg; */
}

std::ostream &RecursiveLogger::spew() {
  throw NotImplemented();
  /* Realm::LoggerMessage msg = this->logger.spew(); */
  /* this->print_prefix(msg); */
  /* return msg; */
}

void RecursiveLogger::print_prefix(std::ostream &msg) const {
  throw NotImplemented();
  /* msg << this->depth << " "; */
  /* for (int i = 0; i < this->depth; i++) { */
  /*   msg << " "; */
  /* } */
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
