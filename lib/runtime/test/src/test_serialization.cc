#include "doctest/doctest.h"
#include "legion/legion_utilities.h"
#include "serialization.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ffconst.h"

using namespace FlexFlow;

TEST_CASE("Serialization") {
  Legion::Serializer sez ();
  Legion::Deserializer dez (sez.get_buffer(), sez.get_buffer_size());

  /* Linear */
  LinearAttrs linear_attrs ();
  Serialization<LinearAttrs> linear;
  linear.serialize(sez, linear_attrs);
  CHECK(linear_attrs == linear.deserialize(dez));

  /* Conv2d */
  Conv2DAttrs conv2d_attrs ();
  Serialization<Conv2DAttrs> conv2d;
  conv2d.serialize(sez, conv2d_attrs);
  CHECK(conv2d_attrs == conv2d.deserialize(dez));
}