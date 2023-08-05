#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
std::vector<int> generateRandomNumbers(const int& k,const int &n){
    // Create a vector to store numbers from 0 to n-1
    std::vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        numbers[i] = i;
    }

    // Shuffle the vector using Fisher-Yates algorithm
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    
    return std::vector<int>(numbers.begin(), numbers.begin() + k);
}