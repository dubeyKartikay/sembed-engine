#include <cstdint>
#include <vector>
#include <unordered_set>
#ifndef SET
#define SET
class Set{
  private:
  std::vector<int64_t> m_data;
  std::unordered_set<int64_t> m_unique_data;
  public:
  Set() = default;
  Set(std::vector<int64_t> & vec);
  void lazyInsert(int64_t val);
  void insert(int64_t val);
};
#endif // !SET
