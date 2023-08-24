#include "Set.hpp"
#include <algorithm>
Set::Set(std::vector<int> & vec) : m_data(vec.size(),0),m_unique_data(vec.begin(),vec.end()){
  std::copy_n(vec.data(),vec.size(),m_data.data()); 
}
void lazyInsert(int val){

}
