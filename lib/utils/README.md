# utils

## visitable

[!WARNING]
`visitable` is deprecated, new code should instead use `dtgen`

### Motivation

FlexFlow's codebase makes heavy use of "plain old data"[^2] types[^1] (referred to as _product types_ in the rest of this document) such as the following:
```cpp
struct Person {
  std::string first_name;
  std::string last_name;
  int age;
};
```
However, this standard implementation defines a set of behaviors that we, the FlexFlow developers, find undesirable:

1. Partial constructibility: for many product types partial constructibility can make code bug-prone. For example, let us consider the following valid code:
```cpp
struct Person {
  Person() = delete;

  std::string first_name;
  std::string last_name;
  int age = 0;
};

Person p{"donald", "knuth"};
```
This code will compile just fine, but will silently use a nonsensical value of `age`. 
Even worse, let us imagine that in the someone else adds an additional field `is_male`. 
Unless they find and update every place in which `Person` is constructed, they will be left with the following code, which 
compiles without errors but is (as of writing this) incorrect.
```cpp
struct Person {
  Person() = delete;

  std::string first_name;
  std::string last_name;
  int age = 0;
  bool is_male = false;
};

Person p{"donald", "knuth", 85};
```

Not only can single fields be undefined/invalid, but whole structs can silently be filled with incorrect values if default constructibility is enabled:
```cpp
Person some_function() {
  Person p;
  if (...) {
    p = {"donald", "knuth", 85};
  }
  return p;
}
```
If the `if` branch is not taken, we will return a `Person` with nonsensical values, as there do not exist any values that naturally form a default.
We could initalize the values as follows
```cpp
struct Person {
  std::string first_name; // initializes to ""
  std::string last_name; // initializes to ""
  int age = 0;
}
```
but this is a completely useless value, and if it shows up anywhere in our code it's probably a bug, since a nameless, age 0 person is probably not a helpful value to have.

3. For product types, `operator==` and `operator!=` are trivial, but still have to be written and maintained, and can easily lead to bugs. For example, 
```
struct Person {
  Person() = delete;
  Person(std::string const &first_name, 
         std::string const &last_name, 
         int age)
    : first_name(first_name),
      last_name(last_name),
      age(age),
    { }

  friend bool operator==(Person const &lhs, Person const &rhs) {
    return lhs.first_name == rhs.first_name 
      && lhs.last_name == rhs.last_name
      && lhs.age == rhs.age;
  }

  friend bool operator!=(Person const &lhs, Person const &rhs) {
    return lhs.first_name != rhs.first_name 
      || lhs.last_name != rhs.last_name
      || lhs.age != rhs.age;
  }

  std::string first_name;
  std::string last_name;
  int age;
};
```
If we take the previous example of adding an additional `is_male` field to `Person`, it can be easy to miss a location, leading to incorrectness. 
For example, we could quite easily end up with
```cpp
struct Person {
  Person() = delete;
  Person(std::string const &first_name, 
         std::string const &last_name, 
         int age,
         bool is_male)
    : first_name(first_name),
      last_name(last_name),
      age(age),
      is_male(is_male)
    { }

  friend bool operator==(Person const &lhs, Person const &rhs) {
    return lhs.first_name == rhs.first_name 
      && lhs.last_name == rhs.last_name
      && lhs.age == rhs.age
      && lhs.is_male == rhs.is_male;
  }

  friend bool operator!=(Person const &lhs, Person const &rhs) {
    return lhs.first_name != rhs.first_name 
      || lhs.last_name != rhs.last_name
      || lhs.age != rhs.age;
      // oops, forgot to update with the new is_male field. Have fun debugging :P
  }

  std::string first_name;
  std::string last_name;
  int age;
  bool is_male;
};
```
and for product types with more fields this grows increasingly tedious to write and maintain. 

4. Hashing: hashing product types is relatively trivial, as long as each of the fields is hashable. But again, we have to do a bunch of extra work to specify this, and this work has to be done for each product type in the codebase.
(**Note:** from here on the examples are growing to grow increasingly long to emphasize the amount of code that needs to be written. Feel free to skim these longer code snippets if you trust the statement that implementing product types in vanilla C++ is tedious)
```cpp
struct Person {
  Person() = delete;
  Person(std::string const &first_name, 
         std::string const &last_name, 
         int age,
         bool is_male)
    : first_name(first_name),
      last_name(last_name),
      age(age),
      is_male(is_male)
    { }

  friend bool operator==(Person const &lhs, Person const &rhs) {
    return lhs.first_name == rhs.first_name 
      && lhs.last_name == rhs.last_name
      && lhs.age == rhs.age
      && lhs.is_male == rhs.is_male;
  }

  friend bool operator!=(Person const &lhs, Person const &rhs) {
    return lhs.first_name != rhs.first_name 
      || lhs.last_name != rhs.last_name
      || lhs.age != rhs.age
      || lhs.is_male != rhs.is_male;
  }

  std::string first_name;
  std::string last_name;
  int age;
  bool is_male;
};

// BEGIN new code
namespace std {

template <>
struct hash<::Person> {
  size_t operator()(::Person const &p) const {
    size_t result = 0;
    hash_combine(result, p.first_name);
    hash_combine(result, p.last_name);
    hash_combine(result, p.age);
    hash_combine(result, p.is_male);
  }
};
// END new code

}
```
and if we also want to support `std::set<Person>` (which requires `Person` to be ordered), we also have to add `operator<`
```cpp
struct Person {
  Person() = delete;
  Person(std::string const &first_name, 
         std::string const &last_name, 
         int age,
         bool is_male)
    : first_name(first_name),
      last_name(last_name),
      age(age),
      is_male(is_male)
    { }

  friend bool operator==(Person const &lhs, Person const &rhs) {
    return lhs.first_name == rhs.first_name 
      && lhs.last_name == rhs.last_name
      && lhs.age == rhs.age
      && lhs.is_male == rhs.is_male;
  }

  friend bool operator!=(Person const &lhs, Person const &rhs) {
    return lhs.first_name != rhs.first_name 
      || lhs.last_name != rhs.last_name
      || lhs.age != rhs.age
      || lhs.is_male != rhs.is_male;
  }

// BEGIN new code
  friend bool operator<(Person const &lhs, Person const &rhs) {
    return lhs.first_name < rhs.first_name 
      || lhs.last_name < rhs.last_name
      || lhs.age < rhs.age
      || lhs.is_male < rhs.is_male;
  }
// END new code

  std::string first_name;
  std::string last_name;
  int age;
  bool is_male;
};

namespace std {

template <>
struct hash<::Person> {
  size_t operator()(::Person const &p) const {
    size_t result = 0;
    hash_combine(result, p.first_name);
    hash_combine(result, p.last_name);
    hash_combine(result, p.age);
    hash_combine(result, p.is_male);
  }
};

}
```
Even for such a simple datatype, we have a significant amount of code that must be written and maintained.
FlexFlow's codebase contains tens if not hundreds of these product types, and so the approach above is infeasible.

[^1]: aka product types, aka Haskell's `data`. Essentially types that are just a tuple of fields with names.
[^2]: by "plain old data" we refer to the general idea behind [C++'s POD](https://en.cppreference.com/w/cpp/named_req/PODType), but not its exact definition

### Adding new visitable types

FlexFlow's `visitable` support provides an easy way to express product types, and prevents any of the bugs listed above.
To express the above definition of `Person` using `visitable`, we would write the following code:
```cpp
struct Person {
  std::string first_name;
  std::string last_name;
  int age;
  req<bool> is_male;
};
FF_VISITABLE_STRUCT(Person, first_name, last_name, age, is_male);
```
The key addition here is the calling the `FF_VISITABLE_STRUCT` macro. 
In addition to defining all of the above functions, this macro also performs a series of compile-time checks (via `static_assert`) to check that the product type is implemented correctly (for example, it will check that the type is not default constructible[^3]).
The only additional change is the addition of the `req` (which stands for `required`) wrapper on the last field. 
Conceptually, `req` is simple: it removes default constructibility of the type it wraps (if the last field in the struct is already not default-constructible, no `req` is needed).
Don't worry if you forget to add a `req`: `FF_VISITABLE_STRUCT` will check that your type properly disables default and partial construction (see [Macro Reference](#macro-reference)).
Combined with [aggregate initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization), we are able to construct a `Person` as follows:
```cpp
Person p = { "donald", "knuth", 85, true };
```
and any subset of the fields would raise an error at compile time. Without any additional code, `Person` supports `operator==`, `operator!=`, `std::hash`, and `operator<`, as well as other more specific features (e.g., [JSON serialization](#json-serialization))

[^3]: The full list of properties is detailed in [Macros Details](#macro-reference)

### Limitations

`visitable` types have two primary limitations. First, they do not support initialization with `(...)`:
```cpp
Person p{ "donald", "knuth", 85, true }; // CORRECT
Person p2("robert", "tarjan", 75, true); // ERROR
```
Secondly, template types cannot be visitable (we hope to remove this limitation in the distant future), but instantiations of them can.
```cpp
template <typename T>
struct MyLists {
  std::vector<T> list1;
  req<std::vector<T>> list2;
};
FF_VISITABLE_STRUCT(MyLists, list1, list2); // ERROR

using MyInts = MyLists<int>;

FF_VISITABLE_STRUCT(MyInts, list1, list2); // CORRECT
```
A smaller limitation is that `FF_VISITABLE_STRUCT` only works from within the `FlexFlow` namespace (this is not much of an issue as all of the `FlexFlow` code resides in a single namespace).

### Advanced Features

While `FF_VISITABLE_STRUCT` matches the behavior of many product types in FlexFlow's codebase, there are exceptions. Many of these resemble the code below:
```cpp
struct Cow { ... };

struct TownPopulation {
  std::vector<Person> people;
  std::vector<Cow> cows;
};
```
Unlike in the `Person` example, `TownPopulation` has an obvious default value: an empty town (i.e., both people and cow are empty).
However, if we write
```cpp
FF_VISITABLE_STRUCT(TownPopulation, people, cows); // ERROR: TownPopulation should not be default constructible
```
we get the something approximating the error in the comment.
If we were to abandon `visitable` entirely, we would have to write (**Note:** long code example to demonstrate how tedious this is, feel free to skim)
```cpp
struct Cow { ... };

struct TownPopulation {
  TownPopulation() = default;
  TownPopulation(std::vector<Person> const &people,
                 std::vector<Cow> const &cows)
    : people(people), 
      cows(cows)
  { }

  friend bool operator==(TownPopulation const &lhs, TownPopulation const &rhs) {
    return lhs.people == rhs.people 
      && lhs.cows == rhs.cows;
  }

  friend bool operator!=(TownPopulation const &lhs, TownPopulation const &rhs) {
    return lhs.people != rhs.people
      || lhs.cows != rhs.cows;
  }

  friend bool operator<(TownPopulation const &lhs, TownPopulation const &rhs) {
    return lhs.people < rhs.people
      || lhs.cows < rhs.cows;
  }

  std::vector<Person> people;
  std::vector<Cow> cows;
};

namespace std {

template <>
struct hash<::TownPopulation> {
  size_t operator()(::TownPopulation const &t) const {
    size_t result = 0;
    hash_combine(result, t.people);
    hash_combine(result, t.cows);
    return result;
  }
};

}
```
which is tedious and bug-prone.
To remove the constructibility checks performed by `FF_VISITABLE_STRUCT`, we simply use `FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION` instead:
```cpp
struct TownPopulation {
  TownPopulation() = default;
  TownPopulation(std::vector<Person> const &people,
                 std::vector<Cow> const &cows)
    : people(people), 
      cows(cows)
  { }

  std::vector<Person> people;
  std::vector<Cow> cows;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(TownPopulation, people, cows);
```
This is also useful for defining structs with specific non-standard constructor signatures. For example,
```cpp
struct TownPopulation {
  TownPopulation() = default;

  // constructs a TownPopulation filled with the given number of random people and cows
  TownPopulation(int num_people,
                 int num_cows)
    : people(generate_random_people_of_size(num_people)),
      cows(generate_random_cows_of_size(num_cows))
  { }

  TownPopulation(std::vector<Person> const &people,
                 std::vector<Cow> const &cows)
    : people(people), 
      cows(cows)
  { }

  std::vector<Person> people;
  std::vector<Cow> cows;
};
```

#### JSON Serialization

TODO

### Macro Reference

The properties that are checked by each macro are as follows:

1. `FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(TYPENAME, ...fields...)`
  - If `length(fields) > 0`:
    - Every field in `TYPENAME` is `std::hash`able
    - Every field in `TYPENAME` is listed under `fields`
    - `TYPENAME` is copy constructible
    - `TYPENAME` is move constructible
    - `TYPENAME` is copy assignable
    - `TYPENAME` is move assignable
    - Every field in `TYPENAME` supports `operator==` 
    - Every field in `TYPENAME` supports `operator!=`
  - If `length(fields) == 0`:
    - `TYPENAME` is copy constructible
    - `TYPENAME` is move constructible
    - `TYPENAME` is copy assignable
    - `TYPENAME` is move assignable

2. `FF_VISITABLE_STRUCT(TYPENAME, ...fields...)` (in addition to the checks in `FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION`)
  - If `length(fields) > 0`:
    - `TYPENAME` is only constructible when all fields are passed in[^4] 
  - If `length(fields) == 0`:
    - `TYPENAME` is default constructible

[^4]: This is usually resolved by either wrapping the last field in a `req` or using `FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION`

### Internals

TODO

## stack_vector, stack_string, stack_map

## strong_typedef

## containers

## graph

## bidict

## type_traits

## test_types
