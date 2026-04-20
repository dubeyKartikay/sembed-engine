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
  static std::mt19937 g(std::random_device{}());
  std::shuffle(perm.begin(), perm.end(), g);
  return perm;
}

std::vector<int> randomSamplingMethod(const int &k, const int &n,int blackList = -1) {
  std::vector<int> numbers;
  static std::mt19937 gen(std::random_device{}());
  std::unordered_set<int> blackListSet;
  if (blackList != -1) {
    blackListSet.insert(blackList);
  }
  while (numbers.size() < k) {
    int number = gen() % n;
    if (blackListSet.count(number) == 0) {
      numbers.push_back(number);
    }
  }
  return numbers;
}

std::vector<int> generateRandomNumbers(const int &k, const int &n,
                                       int blackList = -1) {
  if(n <0 ||k <= 0){
    return {};
  }
  
  if(k < n/3){
    return randomSamplingMethod(k,n,blackList);
  }

  auto perm =  getPermutation(n);
  if(blackList != -1){
    perm.erase(std::remove(perm.begin(), perm.end(), blackList), perm.end());
  }
  if(perm.size() > k){
    perm.resize(k);
  }
  return perm;
}

int getRandomNumber(int start, int end) {
  static std::mt19937 gen(std::random_device{}());
  if (start > end) return start;
  return start + gen() % (end - start + 1);
}
