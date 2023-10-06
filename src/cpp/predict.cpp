
#define FMT_HEADER_ONLY

#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <ATen/ops/nonzero.h>
#include <Magick++.h>
#include <fmt/core.h>
#include <getopt.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include "model_utils.h"
#include "rest/oatpp_entry.h"
#include "utils.h"

using namespace std;
using json = nlohmann::json;

volatile sig_atomic_t ev_flag = 0;
std::atomic<uint32_t> inference_interval_ms = 60000;
vector<string> model_ids;
json settings;
mutex image_queue_mtx, ext_program_mtx, swagger_mtx, models_mtx, model_ids_mtx;
deque<vector<char>> image_queue;
constexpr ssize_t gif_frame_count = 36;
constexpr ssize_t inference_batch_size = 24;
constexpr ssize_t pre_detection_size = 4;
constexpr ssize_t image_queue_min_len =
    pre_detection_size + inference_batch_size + gif_frame_count;
constexpr ssize_t image_queue_max_len = image_queue_min_len * 4;
string cuda_device_string = "cuda:0";
string torch_script_serialization;
vector<torch::jit::script::Module> models;
std::unordered_map<uint32_t, PercentileTracker<float>> pt_dict;

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

void zeromq_ev_loop() {
  spdlog::info("zeromq_ev_loop() started");

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  string zmq_address = settings.value("/inference/zeromq_address"_json_pointer,
                                      "tcp://127.0.0.1:4240");
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ client connected to {}", zmq_address);
  int timeout = 5000;
  subscriber.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  while (!ev_flag) {
    zmq::message_t message;
    try {
      subscriber.recv(&message);
    } catch (const zmq::error_t &e) {
      if (e.num() == EAGAIN) {
        spdlog::warn("subscriber.recv() timed out, will retry");
        continue;
      } else {
        spdlog::error("subscriber.recv() throws exception: {}", e.what());
        continue;
      }
    }
    auto t = vector<char>(message.size());
    if (message.size() == 0) {
      spdlog::warn("ZeroMQ received empty message");
      continue;
    }
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

void execute_external_program_async() {
  auto f = [](string cmd) {
    if (ext_program_mtx.try_lock()) {
      spdlog::info("Calling external program: {}", cmd);
      system(cmd.c_str());
      ext_program_mtx.unlock();
    } else {
      spdlog::warn("ext_program_mtx is locked already, meaning that another {} "
                   "instance is already running",
                   cmd);
    }
  };
  thread th_exec(f, settings["inference"]["on_detected"]["external_program_cpp"]
                        .get<string>());
  th_exec.detach();
}

bool handle_inference_results(vector<at::Tensor> &outputs, at::Tensor &output,
                              const vector<vector<char>> &jpegs) {
  ostringstream oss_raw_result, oss_avg_result;

  for (size_t i = 0; i < outputs.size(); ++i) {
    oss_raw_result << outputs[i];
  }
  oss_avg_result << output;
  at::Tensor y_pred = torch::argmax(output, 1);

  auto nonzero_y_preds_idx = torch::nonzero(y_pred);
  if (nonzero_y_preds_idx.sizes()[0] == 0) {
    return false;
  }
  spdlog::warn("Target detected at {}-th frame in a batch of {} frames",
               nonzero_y_preds_idx[0].item<int>(), inference_batch_size, 0);
  spdlog::info("y_pred:              {}",
               tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
  spdlog::info("nonzero_y_preds_idx: {}",
               tensor_to_string_like_pytorch(nonzero_y_preds_idx, 0,
                                             nonzero_y_preds_idx.sizes()[0]));
  spdlog::info("Raw results are:\n{}", oss_raw_result.str());
  spdlog::info("and arithmetic average of raw results is:\n{}",
               oss_avg_result.str());
  spdlog::info("Build GIF animation");
  vector<Magick::Image> frames;
  spdlog::info("jpegs[{}: {}] of {} images will be used to build the GIF "
               "animation",
               nonzero_y_preds_idx[0].item<int>(),
               nonzero_y_preds_idx[0].item<int>() + gif_frame_count,
               jpegs.size());
  auto gif_width =
      settings.value("/inference/on_detected/gif_size/width"_json_pointer,
                     target_img_size.width);
  auto gif_height =
      settings.value("/inference/on_detected/gif_size/height"_json_pointer,
                     target_img_size.height);
  // nonzero_y_preds_idx[0] stores the first non-zero item
  // index in y_pred. Note that y_pred[0] is NOT the result
  // of jpegs[0], but jpegs[pre_detection_size]
  for (int i = 0; i < gif_frame_count; ++i) {
    // One needs to think for a while to understand the
    // offset between jpegs_idx and y_pred's index--their gap
    // is exactly inference_batch_size, which is implied in
    // the line: cv::imdecode(jpegs[pre_detection_size + i],
    // cv::IMREAD_COLOR));
    auto jpegs_idx = nonzero_y_preds_idx[0].item<int>() + i;
    assert(jpegs_idx < jpegs.size());
    frames.emplace_back(
        Magick::Blob(jpegs[jpegs_idx].data(), jpegs[jpegs_idx].size()));
    frames.back().animationDelay(10); // 100 milliseconds (10 * 1/100th)
    frames.back().resize(Magick::Geometry(gif_width, gif_height));
  }
  string gif_path = settings.value(
      "/inference/on_detected/gif_path"_json_pointer, "/tmp/detected.gif");
  spdlog::info("Saving GIF file (size: {}x{}) to [{}]", gif_width, gif_height,
               gif_path);
  Magick::writeImages(frames.begin(), frames.end(), gif_path);
  execute_external_program_async();
  filesystem::path jpg_dir = settings.value(
      "/inference/on_detected/jpegs_directory"_json_pointer, "/");
  spdlog::info("Saving positive images as JPEG files to [{}]",
               jpg_dir.native());
  for (int i = 0; i < nonzero_y_preds_idx.sizes()[0]; ++i) {
    auto jpegs_idx = nonzero_y_preds_idx[i].item<int>() + pre_detection_size;
    assert(jpegs_idx < jpegs.size());
    filesystem::path jpg_path = jpg_dir / (getCurrentDateTimeString() + ".jpg");
    ofstream outFile(jpg_path, ios::binary);
    if (!outFile) {
      spdlog::error("Failed to open the file [{}]", jpg_path.native());
    } else {
      outFile.write(jpegs[jpegs_idx].data(), jpegs[jpegs_idx].size());
      outFile.close();
    }
  }
  return true;
}

pair<vector<at::Tensor>, at::Tensor>
infer_images(vector<torch::jit::script::Module> &models,
             const vector<vector<char>> &jpegs) {
  auto start = chrono::high_resolution_clock::now();
  vector<torch::Tensor> images_tensors_vec(inference_batch_size);
  torch::Tensor images_tensor;
  vector<torch::jit::IValue> input(1);
  for (int i = 0; i < inference_batch_size; ++i) {
    // images_mats[i] = cv::imdecode(received_jpgs[i], cv::IMREAD_COLOR);
    images_tensors_vec[i] = cv_mat_to_tensor(
        cv::imdecode(jpegs[pre_detection_size + i], cv::IMREAD_COLOR));
  }
  images_tensor = torch::stack(images_tensors_vec);

  input[0] = images_tensor.to(cuda_device_string);

  // images_tensor.sizes()[0] stores number of images
  at::Tensor output =
      torch::zeros({images_tensor.sizes()[0], NUM_OUTPUT_CLASSES});
  output = output.to(cuda_device_string);
  vector<at::Tensor> outputs(models.size());
  {
    std::lock_guard<std::mutex> lock(models_mtx);
    for (size_t i = 0; i < models.size(); ++i) {
      outputs[i] = models[i].forward(input).toTensor();
      output += outputs[i];
    }
  }
  auto end = chrono::high_resolution_clock::now();
  {
    lock_guard<mutex> lock(swagger_mtx);
    auto [it, success] = pt_dict.try_emplace(inference_interval_ms,
                                             PercentileTracker<float>(10000));
    pt_dict.at(inference_interval_ms)
        .addNumber(
            (float)chrono::duration_cast<chrono::milliseconds>(end - start)
                .count() /
            inference_batch_size);
  }
  return make_pair(outputs, output);
}

void interruptible_sleep(const size_t sleep_ms) {
  const size_t interruptible_sleep_ms = 1000;
  if (sleep_ms <= interruptible_sleep_ms) {
    this_thread::sleep_for(chrono::milliseconds(sleep_ms));
  } else {
    size_t slept_ms = 0;
    while (slept_ms < sleep_ms && !ev_flag) {
      slept_ms += interruptible_sleep_ms;
      this_thread::sleep_for(chrono::milliseconds(interruptible_sleep_ms));
    }
  }
}

void inference_ev_loop() {
  int delay_ms = 10000;
  spdlog::info("inference_ev_loop() started, waiting for {}ms before start",
               delay_ms);
  // Otherwise we won't have any images of detected object in the GIF
  assert(pre_detection_size < gif_frame_count);
  assert(gif_frame_count > 0);
  assert(inference_batch_size > 0);
  // Initial wait, just to spread out the stress at the beginning of the run
  interruptible_sleep(delay_ms);
  try {
    {
      std::scoped_lock lck{models_mtx, model_ids_mtx};
      models = load_models(model_ids);
    }

  } catch (const c10::Error &e) {
    spdlog::error("Error loading the model: {}", e.what());
    ev_flag = 1;
    spdlog::critical("The program must exit now");
  }

  Magick::InitializeMagick(nullptr);
  vector<vector<char>> received_jpgs(image_queue_min_len);
  vector<cv::Mat> images_mats(inference_batch_size);

  while (!ev_flag) {
    interruptible_sleep(inference_interval_ms);
    {
      lock_guard<mutex> lock(image_queue_mtx);
      if (image_queue.size() < image_queue_min_len) {
        spdlog::warn("image_queue.size() == {} < image_queue_min_len({}), "
                     "waiting for more images before inference can run",
                     image_queue.size(), image_queue_min_len);
        continue;
      }
      for (int i = 0; i < image_queue_min_len; ++i) {
        received_jpgs[i] = image_queue[i];
      }
      for (int i = 0; i < inference_batch_size; ++i) {
        image_queue.pop_front();
      }
    }
    auto rv = infer_images(models, received_jpgs);
    if (handle_inference_results(rv.first, rv.second, received_jpgs)) {
      const size_t cooldown_ms =
          settings.value("/inference/on_detected/cooldown_sec"_json_pointer,
                         120) *
          1000;
      spdlog::info("inference_ev_loop() will sleep for {}ms after detection",
                   cooldown_ms);
      interruptible_sleep(cooldown_ms);
    }
  }
  spdlog::info("inference_ev_loop() exited gracefully");
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
  spdlog::info("Predict daemon exiting");
  return 0;
}
