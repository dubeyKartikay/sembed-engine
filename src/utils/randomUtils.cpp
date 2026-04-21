#include "utils.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <unordered_set>

std::mt19937_64 makeDeterministicRng(uint64_t salt,
                                     std::initializer_list<uint64_t> values) {
  uint64_t seed = salt;
  for (uint64_t value : values) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  }
  return std::mt19937_64(seed);
}

std::mt19937_64 makeDeterministicRng(
    uint64_t salt, std::initializer_list<uint64_t> integer_values,
    std::initializer_list<float> float_values) {
  uint64_t seed = salt;
  for (uint64_t value : integer_values) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  }
  for (float value : float_values) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    seed ^= static_cast<uint64_t>(bits) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
  }
  return std::mt19937_64(seed);
}

NodeList getPermutation(int64_t n, std::mt19937_64 &rng) {
  if (n <= 0) {
    return {};
  }

  const uint64_t count = static_cast<uint64_t>(n);
  NodeList perm(static_cast<size_t>(count), 0);
  for (uint64_t i = 0; i < count; ++i) {
    perm[static_cast<size_t>(i)] = i;
  }
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

NodeList getPermutation(int64_t n) {
  static std::mt19937_64 rng(std::random_device{}());
  return getPermutation(n, rng);
}

namespace {
NodeList randomSamplingMethod(uint64_t k, uint64_t n,
                              std::mt19937_64 &rng,
                              OptionalNodeId blackList = std::nullopt) {
  NodeList numbers;
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
    const NodeId number = dist(rng);
    if (seen.count(number) == 0) {
      numbers.push_back(number);
      seen.insert(number);
    }
  }
  return numbers;
}
}  // namespace

NodeList generateRandomNumbers(uint64_t k, uint64_t n, std::mt19937_64 &rng,
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
    return randomSamplingMethod(k, n, rng, has_blacklisted_value ? blackList
                                                                 : std::nullopt);
  }

  auto perm = getPermutation(n, rng);
  if (has_blacklisted_value) {
    perm.erase(std::remove(perm.begin(), perm.end(), *blackList), perm.end());
  }
  if (perm.size() > static_cast<size_t>(k)) {
    perm.resize(static_cast<size_t>(k));
  }
  return perm;
}

NodeList generateRandomNumbers(uint64_t k, uint64_t n,
                               OptionalNodeId blackList) {
  static std::mt19937_64 rng(std::random_device{}());
  return generateRandomNumbers(k, n, rng, blackList);
}

int64_t getRandomNumber(int64_t start, int64_t end, std::mt19937_64 &rng) {
  if (start > end) return start;
  std::uniform_int_distribution<int64_t> dist(start, end);
  return dist(rng);
}

int64_t getRandomNumber(int64_t start, int64_t end) {
  static std::mt19937_64 rng(std::random_device{}());
  return getRandomNumber(start, end, rng);
}
