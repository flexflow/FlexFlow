#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H

#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct num_points_t : public strong_typedef<num_points_t, int> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(num_points_t);
FF_TYPEDEF_PRINTABLE(num_points_t, "num_points");

struct side_size_t : public strong_typedef<side_size_t, int> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(side_size_t);
FF_TYPEDEF_PRINTABLE(side_size_t, "side_size");

struct StridedRectangleSide : public use_visitable_cmp<StridedRectangleSide> {
public:
  StridedRectangleSide() = delete;
  StridedRectangleSide(num_points_t const &num_points, int stride)
      : num_points(num_points), stride(stride) {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }
  StridedRectangleSide(side_size_t const &num_points, int stride)
      : num_points(num_points), stride(stride) {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }

  num_points_t get_num_points() const;
  side_size_t get_size() const {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }
  int get_stride() const;

  side_size_t at(num_points_t) const {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }
  num_points_t at(side_size_t) const {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }

public:
  num_points_t num_points;
  int stride;
};

struct StridedRectangle : public use_visitable_cmp<StridedRectangle> {
public:
  StridedRectangle() = delete;
  StridedRectangle(std::vector<StridedRectangleSide> const &sides)
      : sides(sides) {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }

  size_t at(FFOrdered<num_points_t> const &) const;
  StridedRectangleSide at(ff_dim_t const &) const;
  size_t num_dims() const {
    // FIXME: Move this definition elsewhere.
    NOT_IMPLEMENTED();
  }

public:
  FFOrdered<StridedRectangleSide> sides;
};
} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::StridedRectangleSide, num_points, stride);
MAKE_VISIT_HASHABLE(::FlexFlow::StridedRectangleSide);

VISITABLE_STRUCT(::FlexFlow::StridedRectangle, sides);
MAKE_VISIT_HASHABLE(::FlexFlow::StridedRectangle);

#endif
