#include "utils/orthotope/orthotope.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("orthotope_contains_coord") {
    Orthotope orthotope = Orthotope{
        {3_p, 1_p},
    };

    SUBCASE("returns true if coord is in orthotope bounds") {
      SUBCASE("smallest allowed coord") {
        OrthotopeCoord coord = OrthotopeCoord{
            {0_n, 0_n},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("largest allowed coord") {
        OrthotopeCoord coord = OrthotopeCoord{
            {2_n, 0_n},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("returns false if coord is out of orthotope bounds") {
      SUBCASE("dim 0") {
        OrthotopeCoord coord = OrthotopeCoord{
            {3_n, 0_n},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("dim 1") {
        OrthotopeCoord coord = OrthotopeCoord{
            {1_n, 1_n},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = false;

        CHECK(result == correct);
      }
    }

    SUBCASE("throws if num dims of coord does not match num dims of the "
            "orthotope") {
      OrthotopeCoord coord = OrthotopeCoord{
          {0_n, 0_n, 0_n},
      };

      CHECK_THROWS(orthotope_contains_coord(orthotope, coord));
    }

    SUBCASE("works if the orthotope is zero-dimensional") {
      Orthotope orthotope = Orthotope{{}};
      OrthotopeCoord coord = OrthotopeCoord{{}};

      bool result = orthotope_contains_coord(orthotope, coord);
      bool correct = true;

      CHECK(result == correct);
    }
  }

  TEST_CASE("orthotope_get_volume") {
    SUBCASE("1d orthotope volume is just dim size") {
      Orthotope input = Orthotope{{8_p}};

      positive_int result = orthotope_get_volume(input);
      positive_int correct = 8_p;

      CHECK(result == correct);
    }

    SUBCASE("multi-dimensional orthotope") {
      Orthotope input = Orthotope{{3_p, 5_p, 1_p, 2_p}};

      positive_int result = orthotope_get_volume(input);
      int correct = 30;

      CHECK(result == correct);
    }

    SUBCASE("zero-dimensional orthotope has volume 1") {
      Orthotope input = Orthotope{{}};

      positive_int result = orthotope_get_volume(input);
      positive_int correct = 1_p;

      CHECK(result == correct);
    }
  }

  TEST_CASE("flatten_orthotope_coord") {
    SUBCASE("one dimension") {
      OrthotopeCoord coord = OrthotopeCoord{{
          4_n,
      }};

      Orthotope orthotope = Orthotope{{
          13_p,
      }};

      nonnegative_int result = flatten_orthotope_coord(coord, orthotope);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("2d tensor is row-major") {
      positive_int num_rows = 5_p;
      positive_int num_cols = 6_p;

      Orthotope orthotope = Orthotope{{
          num_rows,
          num_cols,
      }};

      CHECK(flatten_orthotope_coord(OrthotopeCoord{{0_n, 0_n}}, orthotope) ==
            0_n);
      CHECK(flatten_orthotope_coord(OrthotopeCoord{{1_n, 0_n}}, orthotope) ==
            num_cols);
      CHECK(flatten_orthotope_coord(OrthotopeCoord{{0_n, 1_n}}, orthotope) ==
            1_n);
    }

    SUBCASE("many dimensions") {
      Orthotope orthotope = Orthotope{{
          4_p,
          1_p,
          3_p,
          2_p,
      }};

      OrthotopeCoord coord = OrthotopeCoord{{
          1_n,
          0_n,
          2_n,
          1_n,
      }};

      nonnegative_int result = flatten_orthotope_coord(coord, orthotope);
      nonnegative_int correct = nonnegative_int{1 * 1 + 2 * 2 + 0 * 6 + 1 * 6};

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      Orthotope orthotope = Orthotope{{}};
      OrthotopeCoord coord = OrthotopeCoord{{}};

      nonnegative_int result = flatten_orthotope_coord(coord, orthotope);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("coordinates out of bounds") {
      Orthotope orthotope = Orthotope{{
          1_p,
          3_p,
          2_p,
      }};

      OrthotopeCoord coord = OrthotopeCoord{{
          0_n,
          4_n,
          1_n,
      }};

      CHECK_THROWS(flatten_orthotope_coord(coord, orthotope));
    }

    SUBCASE("dimensions do not match") {
      Orthotope orthotope = Orthotope{{
          4_p,
          1_p,
          3_p,
          2_p,
      }};

      OrthotopeCoord coord = OrthotopeCoord{{
          0_n,
          2_n,
          1_n,
      }};

      CHECK_THROWS(flatten_orthotope_coord(coord, orthotope));
    }
  }

  TEST_CASE("unflatten_orthotope_coord") {
    SUBCASE("one dimension") {
      Orthotope orthotope = Orthotope{{
          13_p,
      }};
      nonnegative_int offset = 4_n;

      OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
      OrthotopeCoord correct = OrthotopeCoord{{
          4_n,
      }};

      CHECK(result == correct);
    }

    SUBCASE("2d tensor is row-major") {
      positive_int num_rows = 5_p;
      positive_int num_cols = 6_p;

      Orthotope orthotope = Orthotope{{
          num_rows,
          num_cols,
      }};

      SUBCASE("origin point") {
        nonnegative_int offset = 0_n;

        OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
        OrthotopeCoord correct = OrthotopeCoord{{
            0_n,
            0_n,
        }};

        CHECK(result == correct);
      }

      SUBCASE("increment in row dimension") {
        nonnegative_int offset = num_cols.nonnegative_int_from_positive_int();

        OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
        OrthotopeCoord correct = OrthotopeCoord{{
            1_n,
            0_n,
        }};

        CHECK(result == correct);
      }

      SUBCASE("increment in column dimension") {
        nonnegative_int offset = 1_n;

        OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
        OrthotopeCoord correct = OrthotopeCoord{{
            0_n,
            1_n,
        }};

        CHECK(result == correct);
      }
    }

    SUBCASE("many dimensions") {
      Orthotope orthotope = Orthotope{{
          4_p,
          1_p,
          3_p,
          2_p,
      }};

      nonnegative_int offset = nonnegative_int{1 * 1 + 2 * 2 + 0 * 6 + 1 * 6};

      OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
      OrthotopeCoord correct = OrthotopeCoord{{
          1_n,
          0_n,
          2_n,
          1_n,
      }};

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      Orthotope orthotope = Orthotope{{}};
      nonnegative_int offset = 0_n;

      OrthotopeCoord result = unflatten_orthotope_coord(offset, orthotope);
      OrthotopeCoord correct = OrthotopeCoord{{}};

      CHECK(result == correct);
    }

    SUBCASE("offset out of bounds") {
      Orthotope orthotope = Orthotope{{
          1_p,
          3_p,
          2_p,
      }};

      nonnegative_int offset = 6_n;

      CHECK_THROWS(unflatten_orthotope_coord(offset, orthotope));
    }
  }
}
