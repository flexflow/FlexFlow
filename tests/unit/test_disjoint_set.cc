#include "flexflow/utils/disjoint_set.h"
#include "gtest/gtest.h"

TEST(disjoint_set, basic) {
  int ctr = 0;
  int a = ctr++, b = ctr++, c = ctr++, d = ctr++, e = ctr++, f = ctr++;

  disjoint_set<int> ds;
  ds.m_union(a, b);
  ds.m_union(b, c);
  ds.m_union(e, f);
  ds.m_union(d, a);

  assert(ds.find(a) == ds.find(b));
  assert(ds.find(a) == ds.find(c));
  assert(ds.find(a) == ds.find(d));
  assert(ds.find(e) == ds.find(f));
  assert(ds.find(e) != ds.find(a));
}
