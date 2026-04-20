#include <cstdint>
#include <string>
#include <vector>
#include "node_types.hpp"
#ifndef UTILS
#define UTILS
#define rowsize(x) (x*sizeof(float) + sizeof(int64_t))
bool isValidPath(const std::string& path);
bool isValidFile(const std::string& path);
NodeList generateRandomNumbers(uint64_t k, uint64_t n,
                               OptionalNodeId blackList = std::nullopt);
NodeList getPermutation(uint64_t n);
int64_t getRandomNumber(int64_t start, int64_t end);
#endif
