#include <deque>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

#include <getopt.h>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#define FMT_HEADER_ONLY
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include "model_utils.h"
#include "rest/App.h"
#include "utils.h"

using namespace std;
using json = nlohmann::json;

volatile sig_atomic_t ev_flag = 0;
std::atomic<uint32_t> prediction_interval_ms = 60000;
json settings;
mutex image_queue_mtx;
deque<vector<char>> image_queue;
const size_t image_queue_min_len = 8;
const size_t image_queue_max_len = image_queue_min_len * 2;

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help,        -h        Display this help and exit\n"
       << "  --config-path, -c        JSON configuration file path" << endl;
}

void parse_arguments(int argc, char **argv, string &config_path) {
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
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

void zeromq_ev_loop() {
  spdlog::info("zeromq_thread started");

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  subscriber.connect(settings["prediction"]["zeromq_address"].get<string>());
  spdlog::info("Connected to {}",
               settings["prediction"]["zeromq_address"].get<string>());
  subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  while (!ev_flag) {
    zmq::message_t message;
    subscriber.recv(&message);
    auto t = vector<char>(message.size());
    memcpy(t.data(), message.data(), t.size());
    {
      lock_guard<mutex> lock(image_queue_mtx);
      image_queue.push_back(t);
      if (image_queue.size() > image_queue_max_len) {
        image_queue.pop_front();
      }
      spdlog::info("image_queue.size(): {}", image_queue.size());
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed gracefully");
}

void prediction_ev_loop() {
  auto models =
      load_models(settings["model"]["torch_script_serialization"].get<string>(),
                  {"0", "1", "2"});
  while (!ev_flag) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(prediction_interval_ms));
    vector<char> img_vec;
    {
      lock_guard<mutex> lock(image_queue_mtx);
      if (image_queue.size() < image_queue_min_len) {
        continue;
      }
      img_vec = image_queue.front();
    }
    cv::Mat decodedImage = cv::imdecode(img_vec, cv::IMREAD_COLOR);

    vector<torch::Tensor> tensor_vec;
    tensor_vec.push_back(cv_mat_to_tensor(decodedImage, cv::Size(426, 224)));
    torch::Tensor images_tensor = torch::stack(tensor_vec);
    vector<torch::jit::IValue> input(1);
    input[0] = images_tensor.to(torch::kCUDA);
    at::Tensor output =
        torch::zeros({images_tensor.sizes()[0], NUM_OUTPUT_CLASSES});
    output = output.to(torch::kCUDA);
    vector<at::Tensor> outputs(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
      outputs[i] = models[i].forward(input).toTensor();
      output += outputs[i];
    }
    ostringstream oss;
    oss << output;

    spdlog::info("Raw results from {} models are:", models.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      oss.str("");
      oss << outputs[i];
      spdlog::info("\n{}", oss.str());
    }
    oss.str("");
    oss << output;
    spdlog::info("and arithmetic average of raw results is:\n{}", oss.str());
    at::Tensor y_pred = torch::argmax(output, 1);
    oss.str("");
    oss << y_pred;
    spdlog::info("y_pred: {}",
                 tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
  }
}

int main(int argc, char **argv) {
  install_signal_handler();
  string config_path;
  parse_arguments(argc, argv, config_path);
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %5t | %v");
  spdlog::info("Predict daemon started");
  spdlog::info("Loading configurations from {}:", config_path);
  ifstream f(config_path);
  settings = json::parse(f);
  spdlog::info("{}", settings.dump(2));
  initialize_rest_api(
      settings["prediction"]["swagger"]["host"].get<string>(),
      settings["prediction"]["swagger"]["port"].get<int>(),
      settings["prediction"]["swagger"]["advertised_host"].get<string>());
  thread thread_zeromq(zeromq_ev_loop);
  thread thread_prediction(prediction_ev_loop);
  thread_zeromq.join();
  thread_prediction.join();
  finalize_rest_api();
  spdlog::info("Predict daemon exiting");
  return 0;
}
