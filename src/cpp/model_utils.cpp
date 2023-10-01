#include <regex>

#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>

#include "model_utils.h"

using namespace std;

vector<torch::jit::script::Module>
load_models(const string &torch_script_serialization,
            vector<string> model_ids) {
  vector<torch::jit::script::Module> models;
  for (size_t i = 0; i < model_ids.size(); ++i) {
    string model_path = regex_replace(torch_script_serialization,
                                      regex("\\{id\\}"), model_ids[i]);
    spdlog::info("Desearilizing {}-th model from {}", i, model_path);
    try {
      models.emplace_back(torch::jit::load(model_path));
      models[i].to(torch::kCUDA);
      models[i].eval();
    } catch (const c10::Error &e) {
      spdlog::error("Error loading the model: {}", e.what());
    }
  }
  return models;
}