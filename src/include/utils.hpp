#ifndef UTILS
#define UTILS

#include <cstdint>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "node_types.hpp"

#define rowsize(x) (x*sizeof(float) + sizeof(int64_t))
bool isValidPath(const std::string &path);
bool isValidFile(const std::string &path);
std::mt19937_64 makeDeterministicRng(uint64_t salt,
                                     std::initializer_list<uint64_t> values);
std::mt19937_64 makeDeterministicRng(
    uint64_t salt, std::initializer_list<uint64_t> integer_values,
    std::initializer_list<float> float_values);
NodeList generateRandomNumbers(uint64_t k, uint64_t n, std::mt19937_64 &rng,
                               OptionalNodeId blackList = std::nullopt);
NodeList generateRandomNumbers(uint64_t k, uint64_t n,
                               OptionalNodeId blackList = std::nullopt);
NodeList getPermutation(int64_t n, std::mt19937_64 &rng);
NodeList getPermutation(int64_t n);
int64_t getRandomNumber(int64_t start, int64_t end, std::mt19937_64 &rng);
int64_t getRandomNumber(int64_t start, int64_t end);

#endif  // UTILS
