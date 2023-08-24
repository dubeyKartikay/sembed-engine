#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
std::vector<int> getPermutation(int n) { 
  std::vector<int> perm(n,0);
  for (int i = 0; i < n; i++) {
    perm[i] = i;
  }
  std::mt19937 g(100);
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}
std::vector<int> generateRandomNumbers(const int &k, const int &n,
                                       int blackList = -1) {

  std::unordered_set<int> in;
  std::vector<int> vec;
  for (size_t i = 0; i < k; i++) {
    int newR = rand() % n;
    if (newR == blackList) {
      newR = rand() % n;
    }
    if (in.count(newR) != 0) {
      newR = rand() % n;
    }
    if (in.count(newR) != 0) {
      newR = rand() % n;
    }
    if (in.count(newR) != 0) {
      newR = rand() % n;
    }
    if (in.count(newR) == 0 && newR != blackList) {
      vec.push_back(newR);
      in.insert(newR);
    }
  }
  return vec;
}

int getRandomNumber(int start, int end) {
  std::mt19937 gen;
  gen.seed(2);
  return start + gen() % (end - start + 1);
}
