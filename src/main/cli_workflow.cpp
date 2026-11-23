#include "cli_workflow.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <sembed/sembed.hpp>

namespace {

using Json = nlohmann::json;
using Options = std::map<std::string, std::string>;

Options readOptions(int argc, char** argv) {
  Options options;
  for (int i = 2; i < argc; ++i) {
    std::string name = argv[i];
    options[name] = name == "--stdin-jsonl" ? "" : argv[++i];
  }
  return options;
}

uint64_t integer(const Options& options, const std::string& name,
                 uint64_t fallback) {
  return options.count(name) ? std::stoull(options.at(name)) : fallback;
}

float decimal(const Options& options, const std::string& name, float fallback) {
  return options.count(name) ? std::stof(options.at(name)) : fallback;
}

Json search(const sembed::Index& index, const Json& vectorJson,
            const sembed::QueryConfig& config) {
  const auto vector = vectorJson.get<std::vector<float>>();
  Json results = Json::array();
  for (const auto& neighbor :
       sembed::query(index, sembed::FloatVectorView(vector), config)) {
    results.push_back({{"node", neighbor.node},
                       {"record_id", neighbor.recordId},
                       {"distance", neighbor.distance}});
  }
  return {{"results", results}};
}

}  // namespace

int runSembedCli(int argc, char** argv) {
  if (argc < 2 || std::string(argv[argc - 1]).find("help") != std::string::npos) {
    std::cout << "sembed index --dataset PATH --output PATH [options]\n"
                 "sembed query --index PATH (--vector JSON | --stdin-jsonl) "
                 "[options]\n";
    return argc < 2;
  }

  try {
    const std::string command = argv[1];
    const Options options = readOptions(argc, argv);

    if (command == "index") {
      const sembed::IndexConfig config{
          integer(options, "--degree-threshold", 64),
          integer(options, "--search-list-size", 100),
          decimal(options, "--distance-threshold", 1.2F)};
      const std::string output = options.at("--output");
      sembed::Index index = sembed::buildIndex(options.at("--dataset"), config);
      index.save(output);
      std::cout << Json({{"command", "index"},
                         {"index", output},
                         {"records", index.size()},
                         {"dimensions", index.dimensions()},
                         {"config",
                          {{"degree_threshold", config.degreeThreshold},
                           {"search_list_size", config.searchListSize},
                           {"distance_threshold", config.distanceThreshold}}}})
                       .dump(2)
                << '\n';
      return 0;
    }

    const sembed::QueryConfig config{
        integer(options, "--k", 10),
        integer(options, "--search-list-size", 0)};
    const std::string indexPath = options.at("--index");
    const sembed::Index index = sembed::Index::load(indexPath);
    if (options.count("--stdin-jsonl")) {
      for (std::string line; std::getline(std::cin, line);) {
        const Json request = Json::parse(line);
        Json response = search(index, request.at("vector"), config);
        response["id"] = request.at("id");
        std::cout << response.dump() << '\n' << std::flush;
      }
    } else {
      Json response = search(index, Json::parse(options.at("--vector")), config);
      response["command"] = "query";
      response["index"] = indexPath;
      std::cout << response.dump(2) << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sembed: " << error.what() << '\n';
    return 1;
  }
}
