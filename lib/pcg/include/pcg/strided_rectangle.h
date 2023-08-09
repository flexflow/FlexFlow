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

struct side_size_t : public strong_typedef<side_size_t, int> {
  using strong_typedef::strong_typedef;
};

struct StridedRectangleSide : public use_visitable_cmp<StridedRectangleSide> {
public:
  StridedRectangleSide() = delete;
  StridedRectangleSide(num_points_t const &, int stride);
  StridedRectangleSide(side_size_t const &, int stride);

  num_points_t get_num_points() const;
  side_size_t get_size() const;
  int get_stride() const;

  side_size_t at(num_points_t) const;
  num_points_t at(side_size_t) const;

public:
  num_points_t num_points;
  int stride;
};

struct StridedRectangle : public use_visitable_cmp<StridedRectangle> {
public:
  StridedRectangle() = delete;
  StridedRectangle(std::vector<StridedRectangleSide> const &);

  size_t at(FFOrdered<num_points_t> const &) const;
  StridedRectangleSide at(ff_dim_t const &) const;
  size_t num_dims() const;

public:
  FFOrdered<StridedRectangleSide> sides;
};
} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::num_points_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::num_points_t, "num_points");

MAKE_TYPEDEF_HASHABLE(::FlexFlow::side_size_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::side_size_t, "side_size");

VISITABLE_STRUCT(::FlexFlow::StridedRectangleSide, num_points, stride);
MAKE_VISIT_HASHABLE(::FlexFlow::StridedRectangleSide);

VISITABLE_STRUCT(::FlexFlow::StridedRectangle, sides);
MAKE_VISIT_HASHABLE(::FlexFlow::StridedRectangle);

#endif
