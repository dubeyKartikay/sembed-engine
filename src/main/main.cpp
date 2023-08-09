#include<iostream>
#include "load_from_binary.hpp"
#include <filesystem>
#include <string>
#include "dataset.hpp"
namespace fs = std::filesystem;
int main(int argc, char ** argv){
    // std::cout << argc << argv[0] << '\n';
    fs::path data = argv[1];
    std::cout << fs::current_path() / data << '\n';
    fs::path dataFilePath = fs::current_path() / data;

        
}
