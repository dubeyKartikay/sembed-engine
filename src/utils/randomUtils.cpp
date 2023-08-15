#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
std::vector<int> generateRandomNumbers(const int &k, const int &n,
                                       int blackList = -1) {
  // Create a vector to store numbers from 0 to n-1
  std::vector<int> numbers;
  numbers.reserve(n);
  for (int i = 0; i < n -1; i++) {
    if(i == blackList){continue;}
    numbers.push_back(i);
  }

  // Shuffle the vector using Fisher-Yates algorithm
  std::mt19937 gen;
  gen.seed(1);
  std::shuffle(numbers.begin(), numbers.end(), gen);

  return std::vector<int>(numbers.begin(), numbers.begin() + k);
}

int getRandomNumber(int start, int end){
  std::mt19937 gen;
  gen.seed(2);
  return start + gen() %(end - start + 1);
}
