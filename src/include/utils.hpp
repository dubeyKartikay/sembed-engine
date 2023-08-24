#include <iostream>
#include <random>
#include <algorithm>
#include<vector>
#ifndef UTILS
#define UTILS
bool isValidPath(const std::string& path);
std::vector<int> generateRandomNumbers(const int &k,const int &n, int blackList = -1);
std::vector<int> getPermutation(int n);
int getRandomNumber(int start, int end);
#endif
