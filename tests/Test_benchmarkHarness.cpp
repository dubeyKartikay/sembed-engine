#include <cstdlib>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string shellQuote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (const char character : path.string()) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  return quoted + "'";
}

TEST(BenchmarkHarnessRuntime, PythonBenchmarkTestsPass) {
  const std::filesystem::path python(SEMBED_TEST_PYTHON);
  const std::filesystem::path sourceDir(SEMBED_TEST_SOURCE_DIR);
  const std::string command =
      shellQuote(python) + " " +
      shellQuote(sourceDir / "tests/test_benchmarks.py");

  EXPECT_EQ(std::system(command.c_str()), 0);
}

}  // namespace
