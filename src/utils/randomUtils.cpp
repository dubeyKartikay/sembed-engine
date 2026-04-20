#include "utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>

NodeList getPermutation(uint64_t n) {
  NodeList perm(static_cast<size_t>(n), 0);
  for (uint64_t i = 0; i < n; ++i) {
    perm[static_cast<size_t>(i)] = i;
  }
  static std::mt19937 g(std::random_device{}());
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}

namespace {
NodeList randomSamplingMethod(uint64_t k, uint64_t n,
                              OptionalNodeId blackList = std::nullopt) {
  NodeList numbers;
  static std::mt19937_64 gen(std::random_device{}());
  std::unordered_set<NodeId> seen;
  if (blackList && *blackList < n) {
    seen.insert(*blackList);
  }
  if (n == 0 || seen.size() >= n) {
    return {};
  }
  const uint64_t available = n - static_cast<uint64_t>(seen.size());
  if (k > available) {
    k = available;
  }
  std::uniform_int_distribution<uint64_t> dist(0, n - 1);
  while (numbers.size() < static_cast<size_t>(k)) {
    const NodeId number = dist(gen);
    if (seen.count(number) == 0) {
      numbers.push_back(number);
      seen.insert(number);
    }
  }
  return numbers;
}
}  // namespace

NodeList generateRandomNumbers(uint64_t k, uint64_t n,
                               OptionalNodeId blackList) {
  if (k == 0 || n == 0) {
    return {};
  }

  const bool has_blacklisted_value = blackList && *blackList < n;
  const uint64_t available = n - (has_blacklisted_value ? 1ULL : 0ULL);

  if (available == 0) {
    return {};
  }

  if (k > available) {
    k = available;
  }

  if (k < available / 3) {
    return randomSamplingMethod(k, n, has_blacklisted_value ? blackList
                                                            : std::nullopt);
  }

  auto perm = getPermutation(n);
  if (has_blacklisted_value) {
    perm.erase(std::remove(perm.begin(), perm.end(), *blackList), perm.end());
  }
  if (perm.size() > static_cast<size_t>(k)) {
    perm.resize(static_cast<size_t>(k));
  }
  return perm;
}

int64_t getRandomNumber(int64_t start, int64_t end) {
  static std::mt19937_64 gen(std::random_device{}());
  if (start > end) return start;
  std::uniform_int_distribution<int64_t> dist(start, end);
  return dist(gen);
}
