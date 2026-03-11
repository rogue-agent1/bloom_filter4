#!/usr/bin/env python3
"""Counting Bloom filter + Scalable Bloom filter.

Standard Bloom filters can't delete. Counting Bloom uses counters instead of bits.
Scalable Bloom auto-grows by chaining filters when capacity is reached.

Usage: python bloom_filter4.py [--test]
"""

import sys, math, hashlib, struct

def _hashes(item, k, m):
    """Generate k hash indices for item in range [0, m)."""
    if isinstance(item, str):
        item = item.encode()
    h1 = int(hashlib.md5(item).hexdigest(), 16)
    h2 = int(hashlib.sha1(item).hexdigest(), 16)
    return [(h1 + i * h2) % m for i in range(k)]

class CountingBloomFilter:
    """Bloom filter with 4-bit counters — supports delete."""
    def __init__(self, capacity, fp_rate=0.01):
        self.capacity = capacity
        self.fp_rate = fp_rate
        self.m = max(1, int(-capacity * math.log(fp_rate) / (math.log(2)**2)))
        self.k = max(1, int(self.m / capacity * math.log(2)))
        self.counters = [0] * self.m
        self.count = 0

    def add(self, item):
        for idx in _hashes(item, self.k, self.m):
            if self.counters[idx] < 15:  # 4-bit max
                self.counters[idx] += 1
        self.count += 1

    def remove(self, item):
        if not self.query(item):
            return False
        for idx in _hashes(item, self.k, self.m):
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        self.count -= 1
        return True

    def query(self, item):
        return all(self.counters[idx] > 0 for idx in _hashes(item, self.k, self.m))

    def __contains__(self, item):
        return self.query(item)

    def __len__(self):
        return self.count

class ScalableBloomFilter:
    """Auto-growing Bloom filter that chains sub-filters."""
    def __init__(self, initial_capacity=1000, fp_rate=0.01, growth=2):
        self.fp_rate = fp_rate
        self.growth = growth
        self.initial_capacity = initial_capacity
        self.filters = []
        self._add_filter()

    def _add_filter(self):
        # Each successive filter has tighter FP rate to maintain overall guarantee
        scale = len(self.filters)
        cap = self.initial_capacity * (self.growth ** scale)
        # Tighten FP: fp_rate * (1/2)^scale so geometric sum < fp_rate
        fp = self.fp_rate * (0.5 ** scale)
        self.filters.append(CountingBloomFilter(cap, fp))

    def add(self, item):
        if self.filters[-1].count >= self.filters[-1].capacity:
            self._add_filter()
        self.filters[-1].add(item)

    def query(self, item):
        return any(f.query(item) for f in self.filters)

    def __contains__(self, item):
        return self.query(item)

    def __len__(self):
        return sum(len(f) for f in self.filters)

    @property
    def num_filters(self):
        return len(self.filters)

# --- Tests ---

def test_counting_basic():
    bf = CountingBloomFilter(1000, 0.01)
    bf.add("hello")
    bf.add("world")
    assert "hello" in bf
    assert "world" in bf
    assert "foo" not in bf
    assert len(bf) == 2

def test_counting_delete():
    bf = CountingBloomFilter(1000, 0.01)
    bf.add("hello")
    assert "hello" in bf
    bf.remove("hello")
    assert "hello" not in bf
    assert len(bf) == 0

def test_counting_fp_rate():
    bf = CountingBloomFilter(1000, 0.05)
    for i in range(1000):
        bf.add(f"item-{i}")
    fp = sum(1 for i in range(10000, 20000) if f"item-{i}" in bf)
    rate = fp / 10000
    assert rate < 0.15, f"FP rate too high: {rate}"

def test_scalable_growth():
    sbf = ScalableBloomFilter(initial_capacity=100, fp_rate=0.01)
    for i in range(500):
        sbf.add(f"item-{i}")
    assert sbf.num_filters > 1, f"Should have grown, has {sbf.num_filters} filters"
    for i in range(500):
        assert f"item-{i}" in sbf, f"Missing item-{i}"

def test_scalable_fp():
    sbf = ScalableBloomFilter(initial_capacity=500, fp_rate=0.05)
    for i in range(1000):
        sbf.add(f"x-{i}")
    fp = sum(1 for i in range(10000, 20000) if f"x-{i}" in sbf)
    rate = fp / 10000
    assert rate < 0.15, f"Scalable FP rate too high: {rate}"

def test_no_false_negatives():
    bf = CountingBloomFilter(500, 0.01)
    items = [f"test-{i}" for i in range(500)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"False negative: {item}"

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_counting_basic()
        test_counting_delete()
        test_counting_fp_rate()
        test_scalable_growth()
        test_scalable_fp()
        test_no_false_negatives()
        print("All tests passed!")
