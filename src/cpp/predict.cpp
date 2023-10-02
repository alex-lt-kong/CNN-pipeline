#include "ATen/ops/nonzero.h"
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
const size_t inference_batch_size = 16;
const size_t image_queue_max_len = inference_batch_size * 4;

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
      while (image_queue.size() > image_queue_max_len) {
        image_queue.pop_front();
      }
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_ev_loop() exited gracefully");
}

void handle_pred_results(vector<at::Tensor> &outputs, at::Tensor &output) {
  ostringstream oss_raw_result, oss_avg_result, oss_y_pred;

  // spdlog::info("Raw results from {} models are:", models.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    oss_raw_result << outputs[i];
    // spdlog::info("\n{}", oss.str());
  }
  oss_avg_result << output;
  // spdlog::info("and arithmetic average of raw results is:\n{}", oss.str());
  at::Tensor y_pred = torch::argmax(output, 1);

  auto nonzero_preds = torch::nonzero(y_pred);
  if (nonzero_preds.sizes()[0] == 0) {
    return;
  }
  spdlog::info("y_pred: {}",
               tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
  spdlog::info("nonzero_preds: {}",
               tensor_to_string_like_pytorch(nonzero_preds, 0,
                                             nonzero_preds.sizes()[0]));
}

pair<vector<at::Tensor>, at::Tensor>
infer_images(const size_t inference_batch_size,
             vector<torch::jit::script::Module> &models,
             const vector<vector<char>> &received_jpgs) {
  vector<torch::Tensor> images_tensors_vec(inference_batch_size);
  torch::Tensor images_tensor;
  vector<torch::jit::IValue> input(1);
  for (int i = 0; i < inference_batch_size; ++i) {
    // images_mats[i] = cv::imdecode(received_jpgs[i], cv::IMREAD_COLOR);
    images_tensors_vec[i] =
        cv_mat_to_tensor(cv::imdecode(received_jpgs[i], cv::IMREAD_COLOR));
  }
  images_tensor = torch::stack(images_tensors_vec);

  input[0] = images_tensor.to(torch::kCUDA);

  // images_tensor.sizes()[0] stores number of images
  at::Tensor output =
      torch::zeros({images_tensor.sizes()[0], NUM_OUTPUT_CLASSES});
  output = output.to(torch::kCUDA);
  vector<at::Tensor> outputs(models.size());
  for (size_t i = 0; i < models.size(); ++i) {
    outputs[i] = models[i].forward(input).toTensor();
    output += outputs[i];
  }
  return make_pair(outputs, output);
}

void prediction_ev_loop() {
  auto models =
      load_models(settings["model"]["torch_script_serialization"].get<string>(),
                  {"0", "1", "2"});
  vector<vector<char>> received_jpgs(inference_batch_size);
  vector<cv::Mat> images_mats(inference_batch_size);

  const size_t interruptible_sleep_ms = 2000;
  while (!ev_flag) {
    if (prediction_interval_ms <= 2000) {
      this_thread::sleep_for(chrono::milliseconds(prediction_interval_ms));
    } else {
      size_t slept_ms = 0;
      while (slept_ms < prediction_interval_ms && !ev_flag) {
        slept_ms += interruptible_sleep_ms;
        this_thread::sleep_for(chrono::milliseconds(interruptible_sleep_ms));
      }
    }

    {
      lock_guard<mutex> lock(image_queue_mtx);
      if (image_queue.size() < inference_batch_size) {
        spdlog::warn("image_queue.size() == {} < inference_batch_size({}), "
                     "waiting for more images before inference can run",
                     image_queue.size(), inference_batch_size);
        continue;
      }
      for (int i = 0; i < inference_batch_size; ++i) {
        received_jpgs[i] = image_queue.front();
        image_queue.pop_front();
      }
    }
    auto rv = infer_images(inference_batch_size, models, received_jpgs);
    handle_pred_results(rv.first, rv.second);
  }
  spdlog::info("prediction_ev_loop() exited gracefully");
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
  prediction_interval_ms =
      settings["prediction"]["initial_prediction_interval_ms"].get<int>();
  thread thread_zeromq(zeromq_ev_loop);
  thread thread_prediction(prediction_ev_loop);
  if (thread_zeromq.joinable()) {
    thread_zeromq.join();
  }
  if (thread_prediction.joinable()) {
    thread_prediction.join();
  }
  finalize_rest_api();
  spdlog::info("Predict daemon exiting");
  return 0;
}
