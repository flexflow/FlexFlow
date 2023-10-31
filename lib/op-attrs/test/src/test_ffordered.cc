#include "test/op-attrs/doctest.h"
#include "op-attrs/dim_ordered.h"

using namespace FlexFlow;

TEST_CASE("FFOrdered Tests") {
    FFOrdered<size_t> dims1 = {1, 2, 3, 4};  // 用示例数据初始化 FFOrdered 对象
    FFOrdered<size_t> dims2 = {4, 3, 2, 1};  // 用不同顺序的示例数据初始化另一个 FFOrdered 对象

    SUBCASE("Size and Accessors") {
        CHECK(dims1.size() == 4);
        CHECK(dims1.num_dims() == 4);
        CHECK(dims1[0] == 1);
        CHECK(dims1.at(2) == 3);
    }

    SUBCASE("Comparison") {
        CHECK(dims1 == dims1);
        CHECK(dims1 != dims2);
        CHECK(dims1 < dims2);
        CHECK(dims2 > dims1);
    }

    SUBCASE("Iterators begin") {
        FFOrdered<size_t>::iterator it = dims1.begin();
        CHECK(*it == 1);
        ++it;
        CHECK(*it == 2);

        FFOrdered<size_t>::const_iterator cit = dims1.cbegin();
        CHECK(*cit == 1);
        ++cit;
        CHECK(*cit == 2);

        FFOrdered<size_t>::reverse_iterator rit = dims1.rbegin();
        CHECK(*rit == 4);
        ++rit;
        CHECK(*rit == 3);

        FFOrdered<size_t>::const_reverse_iterator crit = dims1.crbegin();
        CHECK(*crit == 4);
        ++crit;
        CHECK(*crit == 3);
    }

    SUBCASE("Const Reverse Iterator Tests") {
        FFOrdered<size_t>::const_reverse_iterator crit = dims.crend();  
        CHECK(crit != dims.crbegin());  

        CHECK(*crit == 1); 
        --crit;
        CHECK(*crit == 2);  
        --crit;
        CHECK(*crit == 3);  
        --crit;
        CHECK(*crit == 4);  
        --crit;
        CHECK(crit == dims.begin()); 
    }
}