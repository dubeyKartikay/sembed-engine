#include <cstdint>
#include <iostream>
#include <random>
#include <algorithm>
#include<vector>
#ifndef UTILS
#define UTILS
bool isValidPath(const std::string& path);
bool isValidFile(const std::string& path);
std::vector<int64_t> generateRandomNumbers(uint64_t k, uint64_t n,
                                           int64_t blackList = -1);
std::vector<int64_t> getPermutation(uint64_t n);
int64_t getRandomNumber(int64_t start, int64_t end);
#endif
