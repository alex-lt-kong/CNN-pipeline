#define FMT_HEADER_ONLY

#include "event_loops.h"
#include "global_vars.h"
#include "model_utils.h"
#include "rest/oatpp_entry.h"
#include "utils.h"

#include <fmt/core.h>
#include <getopt.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help,        -h        Display this help and exit\n"
       << "  --config-path, -c        JSON configuration file path" << endl;
}

void parse_arguments(int argc, char **argv, string &config_path) {
  static struct option long_options[] = {
      {"config-path", required_argument, 0, 'c'},
      {"help", optional_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "c:h", long_options, &option_index)) !=
         -1) {
    switch (opt) {
    case 'c':
      if (optarg != NULL) {
        config_path = string(optarg);
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (config_path.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  install_signal_handler();
  string config_path;
  parse_arguments(argc, argv, config_path);
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %5t | %v");
  spdlog::info("inference_service started");
  spdlog::info("Loading configurations from {}:", config_path);
  ifstream f(config_path);
  settings = json::parse(f);
  spdlog::info("{}", settings.dump(2));
  model_ids = settings.value("/inference/initial_model_ids"_json_pointer,
                             vector<string>{"0"});
  torch_script_serialization = settings.value(
      "/model/torch_script_serialization"_json_pointer, string(""));

  initialize_rest_api(
      settings.value("/inference/swagger/host"_json_pointer, "127.0.0.1"),
      settings.value("/inference/swagger/port"_json_pointer, 8000),
      settings.value("/inference/swagger/advertised_host"_json_pointer,
                     "http://127.0.0.1:8000"));
  inference_interval_ms =
      settings.value("/inference/initial_interval_ms"_json_pointer, 60000);
  thread thread_zeromq(zeromq_ev_loop);
  thread thread_inference(inference_ev_loop);
  if (thread_zeromq.joinable()) {
    thread_zeromq.join();
  }
  if (thread_inference.joinable()) {
    thread_inference.join();
  }
  finalize_rest_api();
  spdlog::info("inference_service exiting");
  return 0;
}
