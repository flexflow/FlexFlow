#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/overload.h"
#include "utils/hash-utils.h"
#include "utils/containers/transform.h"
#include "utils/containers/foldl1.h"
#include "utils/containers/foldr1.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

BinarySeriesSplit::BinarySeriesSplit(BinarySPDecompositionTree const &left, BinarySPDecompositionTree const &right) 
  : left_child_ptr{std::make_shared<BinarySPDecompositionTree>(left)}, 
    right_child_ptr{std::make_shared<BinarySPDecompositionTree>(right)}
{ }

bool BinarySeriesSplit::operator==(BinarySeriesSplit const &other) const {
  return this->tie() == other.tie();
}

bool BinarySeriesSplit::operator!=(BinarySeriesSplit const &other) const {
  return this->tie() != other.tie();
}

bool BinarySeriesSplit::operator<(BinarySeriesSplit const &other) const {
  return this->tie() < other.tie();
}

BinarySPDecompositionTree const &BinarySeriesSplit::left_child() const {
  return *this->left_child_ptr;
}

BinarySPDecompositionTree const &BinarySeriesSplit::right_child() const {
  return *this->right_child_ptr;
}

std::tuple<
  BinarySPDecompositionTree const &,
  BinarySPDecompositionTree const &
> BinarySeriesSplit::tie() const {
  return std::tie(this->left_child(), this->right_child());
}

std::string format_as(BinarySeriesSplit const &s) {
  return fmt::format("<BinarySeriesSplit {} {}>", s.left_child(), s.right_child());
}

std::ostream &operator<<(std::ostream &s, BinarySeriesSplit const &x) {
  return (s << fmt::to_string(x)); 
}

BinaryParallelSplit::BinaryParallelSplit(BinarySPDecompositionTree const &left, BinarySPDecompositionTree const &right) 
  : left_child_ptr{std::make_shared<BinarySPDecompositionTree>(left)}, 
    right_child_ptr{std::make_shared<BinarySPDecompositionTree>(right)}
{ }

bool BinaryParallelSplit::operator==(BinaryParallelSplit const &other) const {
  return this->tie() == other.tie();
}

bool BinaryParallelSplit::operator!=(BinaryParallelSplit const &other) const {
  return this->tie() != other.tie();
}

bool BinaryParallelSplit::operator<(BinaryParallelSplit const &other) const {
  return this->tie() < other.tie();
}

BinarySPDecompositionTree const &BinaryParallelSplit::left_child() const {
  return *this->left_child_ptr;
}

BinarySPDecompositionTree const &BinaryParallelSplit::right_child() const {
  return *this->right_child_ptr;
}

std::tuple<
  BinarySPDecompositionTree const &,
  BinarySPDecompositionTree const &
> BinaryParallelSplit::tie() const {
  return std::tie(this->left_child(), this->right_child());
}

std::string format_as(BinaryParallelSplit const &s) {
  return fmt::format("<BinaryParallelSplit {} {}>", s.left_child(), s.right_child());
}

std::ostream &operator<<(std::ostream &s, BinaryParallelSplit const &x) {
  return (s << fmt::to_string(x)); 
}


BinarySPDecompositionTree::BinarySPDecompositionTree(BinarySeriesSplit const &s)
  : root{s}
{ }

BinarySPDecompositionTree::BinarySPDecompositionTree(BinaryParallelSplit const &s)
  : root{s}
{ }

BinarySPDecompositionTree::BinarySPDecompositionTree(Node const &n)
  : root{n}
{ }

bool BinarySPDecompositionTree::operator==(BinarySPDecompositionTree const &other) const {
  return this->tie() == other.tie();
}

bool BinarySPDecompositionTree::operator!=(BinarySPDecompositionTree const &other) const {
  return this->tie() != other.tie();
}

bool BinarySPDecompositionTree::operator<(BinarySPDecompositionTree const &other) const {
  return this->tie() < other.tie();
}

SPDecompositionTreeNodeType BinarySPDecompositionTree::get_node_type() const {
  // implemented using std::visit as opposed to this->visit because 
  // this->visit is implemented using this function, so doing so would
  // create infinite recursion
  return std::visit(overload {
    [](BinarySeriesSplit const &) { return SPDecompositionTreeNodeType::SERIES; },
    [](BinaryParallelSplit const &) { return SPDecompositionTreeNodeType::PARALLEL; },
    [](Node const &) { return SPDecompositionTreeNodeType::NODE; },
  }, this->root);
}

BinarySeriesSplit const &BinarySPDecompositionTree::require_series() const {
  return this->get<BinarySeriesSplit>();
}

BinaryParallelSplit const &BinarySPDecompositionTree::require_parallel() const {
  return this->get<BinaryParallelSplit>();
}

Node const &BinarySPDecompositionTree::require_node() const {
  return this->get<Node>();
}

BinarySPDecompositionTree BinarySPDecompositionTree::series(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
}

BinarySPDecompositionTree BinarySPDecompositionTree::parallel(BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
}

BinarySPDecompositionTree BinarySPDecompositionTree::node(Node const &n) {
  return BinarySPDecompositionTree{n};
}

std::tuple<std::variant<BinarySeriesSplit, BinaryParallelSplit, Node> const &>
  BinarySPDecompositionTree::tie() const {
  return std::tie(this->root);
}

std::string format_as(BinarySPDecompositionTree const &t) {
  return t.visit<std::string>(overload {
    [](BinarySeriesSplit const &s) { return fmt::format("<BinarySPDecompositionTree {}>", s); },
    [](BinaryParallelSplit const &s) { return fmt::format("<BinarySPDecompositionTree {}>", s); },
    [](Node const &n) { return fmt::format("<BinarySPDecompositionTree {}>", n); },
  });
}
std::ostream &operator<<(std::ostream &s, BinarySPDecompositionTree const &t) {
  return (s << fmt::to_string(t));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::BinarySeriesSplit>::operator()(::FlexFlow::BinarySeriesSplit const &s) const {
  return get_std_hash(s.tie());
}

size_t hash<::FlexFlow::BinaryParallelSplit>::operator()(::FlexFlow::BinaryParallelSplit const &s) const {
  return get_std_hash(s.tie());
}

size_t hash<::FlexFlow::BinarySPDecompositionTree>::operator()(::FlexFlow::BinarySPDecompositionTree const &s) const {
  return get_std_hash(s.tie());
}

} // namespace std 

namespace nlohmann {

::FlexFlow::BinarySeriesSplit 
  adl_serializer<::FlexFlow::BinarySeriesSplit>::from_json(json const &j) {

  return ::FlexFlow::BinarySeriesSplit{
    j.at("left_child").template get<::FlexFlow::BinarySPDecompositionTree>(),
    j.at("right_child").template get<::FlexFlow::BinarySPDecompositionTree>(),
  };
};

void adl_serializer<::FlexFlow::BinarySeriesSplit>::to_json(json &j, ::FlexFlow::BinarySeriesSplit const &v) {
  j["__type"] = "BinarySeriesSplit";
  j["left_child"] = v.left_child();
  j["right_child"] = v.right_child();
}

::FlexFlow::BinaryParallelSplit 
  adl_serializer<::FlexFlow::BinaryParallelSplit>::from_json(json const &j) {

  return ::FlexFlow::BinaryParallelSplit{
    j.at("left_child").template get<::FlexFlow::BinarySPDecompositionTree>(),
    j.at("right_child").template get<::FlexFlow::BinarySPDecompositionTree>(),
  };
};

void adl_serializer<::FlexFlow::BinaryParallelSplit>::to_json(json &j, ::FlexFlow::BinaryParallelSplit const &v) {
  j["__type"] = "BinaryParallelSplit";
  j["left_child"] = v.left_child();
  j["right_child"] = v.right_child();
}

::FlexFlow::BinarySPDecompositionTree 
  adl_serializer<::FlexFlow::BinarySPDecompositionTree>::from_json(json const &j) {

  std::string key = j.at("type").get<std::string>();

  if (key == "series") {
    return ::FlexFlow::BinarySPDecompositionTree{
      j.at("value").get<::FlexFlow::BinarySeriesSplit>(),
    };
  } else if (key == "parallel") {
    return ::FlexFlow::BinarySPDecompositionTree{
      j.at("value").get<::FlexFlow::BinaryParallelSplit>(),
    };
  } else if (key == "node") {
    return ::FlexFlow::BinarySPDecompositionTree{
      j.at("value").get<::FlexFlow::Node>(),
    };
  } else {
    throw ::FlexFlow::mk_runtime_error(fmt::format("Unknown json type key: {}", key));
  }
};

void adl_serializer<::FlexFlow::BinarySPDecompositionTree>::to_json(json &j, ::FlexFlow::BinarySPDecompositionTree const &v) {
  j["__type"] = "BinarySPDecompositionTree";
  v.visit<std::monostate>(::FlexFlow::overload {
    [&](::FlexFlow::BinarySeriesSplit const &s) { 
      j["type"] = "series";
      j["value"] = s;
      return std::monostate{};
    },
    [&](::FlexFlow::BinaryParallelSplit const &p) { 
      j["type"] = "parallel";
      j["value"] = p;
      return std::monostate{};
    },
    [&](::FlexFlow::Node const &n) { 
      j["type"] = "node";
      j["value"] = n;
      return std::monostate{};
    },
  });
}

} // namespace nlohmann

namespace rc {


Gen<::FlexFlow::BinarySeriesSplit>
    Arbitrary<::FlexFlow::BinarySeriesSplit>::arbitrary() {
  return gen::withSize([](int size) {
    return gen::mapcat(gen::inRange(1, size), [size](int left_size) {
      int right_size = size - left_size - 1;

      return gen::construct<::FlexFlow::BinarySeriesSplit>(
        gen::resize(left_size, gen::arbitrary<FlexFlow::BinarySPDecompositionTree>()),
        gen::resize(right_size, gen::arbitrary<FlexFlow::BinarySPDecompositionTree>()));
    });
  });
}

Gen<::FlexFlow::BinaryParallelSplit>
    Arbitrary<::FlexFlow::BinaryParallelSplit>::arbitrary() {
  return gen::withSize([](int size) {
    return gen::mapcat(gen::inRange(1, size), [size](int left_size) {
      int right_size = size - left_size - 1;

      return gen::construct<::FlexFlow::BinaryParallelSplit>(
        gen::resize(left_size, gen::arbitrary<FlexFlow::BinarySPDecompositionTree>()),
        gen::resize(right_size, gen::arbitrary<FlexFlow::BinarySPDecompositionTree>()));
    });
  });
}

Gen<::FlexFlow::BinarySPDecompositionTree>
    Arbitrary<::FlexFlow::BinarySPDecompositionTree>::arbitrary() {
  return gen::withSize([](int size) {
    if (size < 3) {
      return gen::construct<::FlexFlow::BinarySPDecompositionTree>(
        gen::construct<::FlexFlow::Node>(
          gen::nonNegative<int>()));
    } else {
      return gen::oneOf(
        gen::construct<::FlexFlow::BinarySPDecompositionTree>(
          gen::arbitrary<::FlexFlow::BinarySeriesSplit>()),
        gen::construct<::FlexFlow::BinarySPDecompositionTree>(
          gen::arbitrary<::FlexFlow::BinaryParallelSplit>()));  
    }
  });
}

} // namespace rc
