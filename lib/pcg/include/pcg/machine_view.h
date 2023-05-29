#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include <vector>
#include <cstddef>
#include <ostream>
#include "utils/visitable.h"
#include "utils/graph.h"
#include "device_id.h"
#include "device_type.h"
#include "strided_rectangle.h"

namespace FlexFlow {


struct MachineView : public use_visitable_cmp<MachineView> {
  MachineView() = delete;
  MachineView(device_id_t const &, StridedRectangle const &);

  std::vector<int> device_ids() const;

  device_id_t at(FFOrdered<num_points_t> const &coord) const;
  StridedRectangleSide at(size_t) const;
public:
  device_id_t start;
  StridedRectangle rect;
};

std::size_t num_dims(MachineView const &);
std::size_t num_devices(MachineView const &);
DeviceType get_device_type(MachineView const &);

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(device_id_t start, num_points_t num_points, int stride = 1);
MachineView make_1d_machine_view(device_id_t start, side_size_t interval_size, int stride = 1);

MachineView make_1d_machine_view(device_id_t start, size_t interval_size);

}

VISITABLE_STRUCT(::FlexFlow::MachineView, start, rect);
MAKE_VISIT_HASHABLE(::FlexFlow::MachineView);

MAKE_TYPEDEF_HASHABLE(::FlexFlow::DeviceID);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::DeviceID, "DeviceID");

namespace fmt {

template <>
struct formatter<::FlexFlow::DeviceType> : formatter<string_view> { 
  template <typename FormatContext>
  auto format(::FlexFlow::DeviceType d, FormatContext& ctx) const -> decltype(ctx.out()) {
    using ::FlexFlow::DeviceType;

    string_view name = "unknown";
    switch (d) {
      case DeviceType::GPU: name = "GPU"; break;
      case DeviceType::CPU: name = "CPU"; break;
    }
    return formatter<string_view>::format(name, ctx);
  } 
};

};

namespace std {
template <>
struct hash<::FlexFlow::StridedRectangle> {
  size_t operator()(::FlexFlow::StridedRectangle const &) const; 
};

template <>
struct hash<::FlexFlow::MachineView> {
  size_t operator()(::FlexFlow::MachineView const &) const;
};

template <>
struct hash<::FlexFlow::MachineResource> {
  size_t operator()(::FlexFlow::MachineResource const &) const;
};
}; // namespace std

#endif // _FLEXFLOW_MACHINE_VIEW_H
#endif 
