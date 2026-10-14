#include "utils/dot/dot_html_table_cell_contents.h"
#include "utils/dot/dot_html_table.dtg.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

DotHtmlTableCellContents::DotHtmlTableCellContents(std::string const &s)
    : value(s) {}

DotHtmlTableCellContents::DotHtmlTableCellContents(DotHtmlTable const &t)
    : value(std::make_shared<DotHtmlTable>(t)) {}

DotHtmlTable const &DotHtmlTableCellContents::require_nested() const {
  return *std::get<std::shared_ptr<DotHtmlTable>>(this->value);
}

std::string const &DotHtmlTableCellContents::require_simple() const {
  return std::get<std::string>(this->value);
}

bool DotHtmlTableCellContents::is_simple() const {
  return std::holds_alternative<std::string>(this->value);
}

bool DotHtmlTableCellContents::is_nested() const {
  return std::holds_alternative<std::shared_ptr<DotHtmlTable>>(this->value);
}

bool DotHtmlTableCellContents::operator==(
    DotHtmlTableCellContents const &other) const {
  if (this->is_simple() && other.is_simple()) {
    return this->require_simple() == other.require_simple();
  } else if (this->is_nested() && other.is_nested()) {
    return this->require_nested() == other.require_nested();
  } else {
    return false;
  }
}

bool DotHtmlTableCellContents::operator!=(
    DotHtmlTableCellContents const &other) const {
  if (this->is_simple() && other.is_simple()) {
    return this->require_simple() != other.require_simple();
  } else if (this->is_nested() && other.is_nested()) {
    return this->require_nested() != other.require_nested();
  } else {
    return true;
  }
}

std::string format_as(DotHtmlTableCellContents const &c) {
  if (c.is_simple()) {
    return c.require_simple();
  } else {
    return fmt::to_string(c.require_nested());
  }
}

std::ostream &operator<<(std::ostream &s, DotHtmlTableCellContents const &x) {
  return (s << fmt::to_string(x));
}

void to_json(nlohmann::json &j, DotHtmlTableCellContents const &c) {
  j["__type"] = "DotHtmlTableCellContents";
  if (c.is_simple()) {
    j["type"] = "simple";
    j["value"] = c.require_simple();
  } else {
    ASSERT(c.is_nested());

    j["type"] = "nested";
    j["value"] = c.require_nested();
  }
}

} // namespace FlexFlow
