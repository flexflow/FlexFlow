#include "test/utils/doctest.h"
#include "utils/json.h"

namespace FlexFlow {

struct Struct0 {
  // No fields. It is unlikely that this will ever happen in FlexFlow, but who
  // knows?
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Struct0);
FF_VISIT_FMTABLE(Struct0);

struct Struct1 {
  char c;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Struct1, c);
FF_VISIT_FMTABLE(Struct1);

struct Struct2 {
  std::string str;
  int i;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Struct2, str, i);
FF_VISIT_FMTABLE(Struct2);

struct Struct3 {
  unsigned long long ll;
  bool b;
  short s;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Struct3, ll, b, s);
FF_VISIT_FMTABLE(Struct3);

struct NotVisitable {
  float f;
  int i;
};

} // namespace FlexFlow

static std::string str(json const &j) {
  std::stringstream ss;
  ss << j;
  return ss.str();
}

template <typename T>
static void serde(T const &v, std::string const &expected) {
  json j = v;
  CHECK(str(j) == expected);
  CHECK(j.get<T>() == v);
}

// This is used to check that the serialization of the struct produces the
// correct fields. It seems as if the fields are not processed in any specific
// order, so we need to check that everything is present and that the
// punctuation is in order. Each element of the list argument will be the
// serialization of one field.
template <typename T>
static void serde(T const& v, std::initializer_list<const char*> list) {
  json j = v;
  std::string ser = str(j);

  CHECK(ser.front() == '{');
  for (const char* f : list)
    CHECK(ser.find(f) != std::string::npos);
  CHECK(ser.back() == '}');

  // The remaining characters should be commas. But checking for them would be
  // a bit more involved since it would need us to look at anything that was not
  // part of the field itself. Instead, just check that the length is what we
  // expect it to be. We start at two for the two braces. There should be
  // list.size() - 1 commas.
  size_t len = 2;
  for (const char* f : list)
    len += strlen(f);
  CHECK(ser.size() == len + list.size() - 1);
}

TEST_CASE("json_type_name") {
  SUBCASE("builtin") {
    // We don't check for the fixed-width types because they are just typedefs
    // for some builtin type. Those mappings are platform dependent, so it
    // doesn't make a great deal of sense to check them.
    CHECK(json_type_name<bool>::name == "bool");

    CHECK(json_type_name<char>::name == "char");
    CHECK(json_type_name<signed char>::name == "signed char");
    CHECK(json_type_name<unsigned char>::name == "unsigned char");

    CHECK(json_type_name<short>::name == "short");
    CHECK(json_type_name<int>::name == "int");
    CHECK(json_type_name<long>::name == "long");
    CHECK(json_type_name<long long>::name == "long long");
    CHECK(json_type_name<unsigned char>::name == "unsigned char");
    CHECK(json_type_name<unsigned short>::name == "unsigned short");
    CHECK(json_type_name<unsigned int>::name == "unsigned int");
    CHECK(json_type_name<unsigned long>::name == "unsigned long");
    CHECK(json_type_name<unsigned long long>::name == "unsigned long long");
    CHECK(json_type_name<float>::name == "float");
    CHECK(json_type_name<double>::name == "double");
    CHECK(json_type_name<long double>::name == "long double");
  }

  SUBCASE("struct") {
    CHECK(json_type_name<FlexFlow::Struct0>::name == "::FlexFlow::Struct0");
    CHECK(json_type_name<FlexFlow::Struct1>::name == "::FlexFlow::Struct1");
    CHECK(json_type_name<FlexFlow::Struct2>::name == "::FlexFlow::Struct2");
    CHECK(json_type_name<FlexFlow::NotVisitable>::name == "unnamed struct");
  }
}

TEST_CASE("struct") {
  serde(Struct0{}, "null");
  serde(Struct1{'A'}, "{\"c\":65}");
  serde(Struct2{"string", 42}, {"\"i\":42","\"str\":\"string\""});
  serde(Struct3{99, false, 12}, {"\"s\":12","\"b\":false","\"ll\":99"});
}

TEST_CASE("req") {
  SUBCASE("char") {
    req<char> r('c');
    json j = json(r);
    CHECK(str(j) == "99");

    // FIXME: There is currently a bug of some sort that causes this to not
    // compile.
    CHECK(j.get<req<char>>() == r);
  }
}

TEST_CASE("optional") {
  SUBCASE("int") {
    serde<optional<int>>(nullopt, "null");
    serde<optional<int>>(11, "11");
  }

  SUBCASE("string") {
    serde<optional<std::string>>(nullopt, "null");
    serde<optional<std::string>>("string", "\"string\"");
  }
}

TEST_CASE("variant") {
  SUBCASE("single primitive") {
    serde(variant<int>(11), "{\"index\":0,\"type\":\"int\",\"value\":11}");
  }

  SUBCASE("multiple primitives") {
    using Type = variant<int, char, bool>;
    serde<Type>(11, "{\"index\":0,\"type\":\"int\",\"value\":11}");
    serde<Type>('f', "{\"index\":1,\"type\":\"char\",\"value\":102}");
    serde<Type>(true, "{\"index\":2,\"type\":\"bool\",\"value\":true}");
    serde<Type>(false, "{\"index\":2,\"type\":\"bool\",\"value\":false}");
  }
}
