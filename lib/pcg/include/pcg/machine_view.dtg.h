// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/machine_view.struct.toml
/* proj-data
{
  "generated_from": "16c571e6bb82d7ef88e5d2a9146638f4"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_VIEW_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_VIEW_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "pcg/device_id_t.dtg.h"
#include "pcg/strided_rectangle.dtg.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct MachineView {
  MachineView() = delete;
  explicit MachineView(::FlexFlow::device_id_t const &start,
                       ::FlexFlow::StridedRectangle const &rect);

  bool operator==(MachineView const &) const;
  bool operator!=(MachineView const &) const;
  bool operator<(MachineView const &) const;
  bool operator>(MachineView const &) const;
  bool operator<=(MachineView const &) const;
  bool operator>=(MachineView const &) const;
  ::FlexFlow::device_id_t start;
  ::FlexFlow::StridedRectangle rect;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::MachineView> {
  size_t operator()(::FlexFlow::MachineView const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::MachineView> {
  static ::FlexFlow::MachineView from_json(json const &);
  static void to_json(json &, ::FlexFlow::MachineView const &);
};
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(MachineView const &);
std::ostream &operator<<(std::ostream &, MachineView const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_VIEW_DTG_H
