#define FMT_HEADER_ONLY

#include "event_loops.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "model_utils.h"
#include "utils.h"

#include <cxxopts.hpp>
#include <fmt/core.h>
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

namespace GV = CnnPipeline::GlobalVariables;
namespace EL = CnnPipeline::EventLoops;

int main(int argc, char **argv) {
  install_signal_handler(&GV::ev_flag);
  string config_path;

  cxxopts::Options options(argv[0], "Inference service");
  // clang-format off
  options.add_options()
    ("h,help", "print help message")
    ("c,config-path", "JSON configuration file path", cxxopts::value<string>()->default_value(config_path));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") || !result.count("config-path")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  config_path = result["config-path"].as<std::string>();

  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %5t | %v");
  spdlog::info("inference_service started (git commit: {})", GIT_COMMIT_HASH);
  spdlog::info("Loading configurations from {}:", config_path);
  ifstream f(config_path);
  GV::settings = json::parse(f);
  spdlog::info("{}", GV::settings.dump(2));
  GV::model_ids = GV::settings.value(
      "/inference/initial_model_ids"_json_pointer, vector<string>{"0"});
  GV::ts_model_path =
      GV::settings.value("/model/ts_model_path"_json_pointer, string(""));
  GV::cuda_device_string =
      GV::settings.value("/inference/cuda_device"_json_pointer, "cuda:0");

  GV::inference_batch_size =
      GV::settings.value("/inference/batch_size"_json_pointer, 24);
  GV::gif_frame_count = GV::pre_detection_size + GV::inference_batch_size +
                        GV::post_detection_size;
  GV::image_queue_size = GV::gif_frame_count * 4;
  GV::snapshot_pc_queue =
      moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>(
          GV::image_queue_size);
  GV::inference_result_pc_queue =
      moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>(
          GV::image_queue_size * 4);

  initialize_rest_api(
      GV::settings.value("/inference/swagger/host"_json_pointer, "127.0.0.1"),
      GV::settings.value("/inference/swagger/port"_json_pointer, 8000),
      GV::settings.value("/inference/swagger/advertised_host"_json_pointer,
                         "http://127.0.0.1:8000"));

  thread thread_zeromq_consumer(EL::zeromq_consumer_ev_loop);
  thread thread_inference(EL::inference_ev_loop);
  thread thread_zero_producer(EL::zeromq_producer_ev_loop);
  if (thread_zeromq_consumer.joinable()) {
    thread_zeromq_consumer.join();
  }
  if (thread_inference.joinable()) {
    thread_inference.join();
  }
  if (thread_zero_producer.joinable()) {
    thread_zero_producer.join();
  }
  finalize_rest_api();
  spdlog::info("inference_service exiting");
  return 0;
}
