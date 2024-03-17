#include "utils/deduplicated_priority_queue/deduplicated_priority_queue.h"
#include "utils/testing.h"
#include <string>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("deduplicated_priority_queue") {
    DeduplicatedPriorityQueue<int> q;
    CHECK(q.empty());
    CHECK(q.size() == 0);

    q.push(1);
    CHECK(q.size() == 1);

    q.push(2);
    CHECK(q.size() == 2);

    q.push(3);
    CHECK(q.size() == 3);

    q.push(2);
    CHECK(q.size() == 3);

    CHECK(q.top() == 3);
    CHECK(q.top() == 3); // check that top does not pop
    q.pop();
    CHECK(q.size() == 2);

    CHECK(q.top() == 2);
    q.pop();
    CHECK(q.size() == 1);

    CHECK(q.top() == 1);
    q.pop();
    CHECK(q.size() == 0);
    CHECK(q.empty());
  }
}
