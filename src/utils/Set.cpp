#include "Set.hpp"
#include <algorithm>
Set::Set(std::vector<int> &vec)
    : m_data(vec.size(), 0), m_unique_data(vec.begin(), vec.end()) {
  std::copy_n(vec.data(), vec.size(), m_data.data());
}
void Set::lazyInsert(int val) {
  if (m_unique_data.count(val) != 0) {
    return;
  }
  m_data.push_back(val);
  m_unique_data.insert(val);
}
void Set::insert(int val) {
  if(m_unique_data.count(val)!=0){
    return;
  }

}

