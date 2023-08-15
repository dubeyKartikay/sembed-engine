#include <iostream>
#include <random>
#include <algorithm>
#include<vector>
#ifndef UTILS
#define UTILS
bool isValidPath(const std::string& path);
std::vector<int> generateRandomNumbers(const int &k,const int &n, int blackList = -1);
int getRandomNumber(int start, int end);
#endif
