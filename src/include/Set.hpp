#include <vector>
#include <unordered_set>
#ifndef SET
#define SET
class Set{
  private:
  std::vector<int> m_data;
  std::unordered_set<int> m_unique_data;
  public:
  Set() = default;
  Set(std::vector<int> & vec);
  void lazyInsert(int val);
  void insert(int val);
};
#endif // !SET
