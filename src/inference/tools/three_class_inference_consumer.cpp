
#define FMT_HEADER_ONLY

#include "../utils.h"
#include "inference_result.pb.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <Magick++.h>
#pragma GCC diagnostic pop
#include <cxxopts.hpp>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include <string>

using namespace std;
using json = nlohmann::json;

static volatile sig_atomic_t ev_flag = 0;
static string zmq_address;
static string external_program;
static size_t queue_size = 128;
constexpr size_t batch_size = 48;
static int gif_width;
static int gif_height;
static string gif_path;
static int gif_frame_interval;
static cv::Size zmq_payload_mat_size;
static filesystem::path jpg_dir;
static int zmq_payload_mat_type = CV_8UC3;
static size_t cooldown_sec;
static vector<int> include_outputs;
static vector<int> exclude_outputs;
static auto inference_result_pc_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>(
        queue_size);

void save_positive_outputs_as_jpeg(const vector<vector<uchar>> &jpegs,
                                   array<int, batch_size> labels) {

  spdlog::info("Saving positive images as JPEG files to [{}]",
               jpg_dir.native());
  for (size_t i = 0; i < batch_size; ++i) {
    if (labels[i] != 2)
      continue;
    filesystem::path jpg_path =
        jpg_dir / (get_current_datetime_string() + ".jpg");
    ofstream out_file(jpg_path, ios::binary);
    if (!out_file) {
      spdlog::error("Failed to open the file [{}]", jpg_path.native());
    } else {
      out_file.write((char *)jpegs[i].data(), jpegs[i].size());
      out_file.close();
    }
  }
}

void execute_external_program() {
  if (external_program.size() == 0) {
    spdlog::warn("external_program unset, not calling...");
    return;
  }
  spdlog::info("Calling external program: {}", external_program);
  (void)!system(external_program.c_str());
}

void prepare_images(deque<InferenceResultMsg> results,
                    array<int, batch_size> labels) {
  vector<vector<uchar>> jpegs(batch_size);
  // cv::imdecode(v, cv::IMREAD_COLOR);
  vector<Magick::Image> frames;
  for (size_t i = 0; i < batch_size; ++i) {
    cv::imencode(".jpg",
                 cv::Mat(zmq_payload_mat_size, zmq_payload_mat_type,
                         (void *)results[i].payload().data()),
                 jpegs[i]);
    frames.emplace_back(Magick::Blob(jpegs[i].data(), jpegs[i].size()));
    frames.back().animationDelay(gif_frame_interval);
    frames.back().resize(Magick::Geometry(gif_width, gif_height));
  }

  spdlog::info("Saving GIF file (size: {}x{}) to [{}]", gif_width, gif_height,
               gif_path);
  Magick::writeImages(frames.begin(), frames.end(), gif_path);
  save_positive_outputs_as_jpeg(jpegs, labels);
}

void inference_handling_ev_loop() {
  spdlog::info("inference_handling_ev_loop() started");
  deque<InferenceResultMsg> results;
  array<int, batch_size> labels;
  int neg_idx = -1;
  int pos_idx = -1;
  int dequed_count = 0;
  auto last_trigger_at = time(0) - cooldown_sec;
  while (!ev_flag) {
    if (inference_result_pc_queue.size_approx() < batch_size - results.size()) {
      interruptible_sleep(1000, &ev_flag);
      continue;
    }
    auto cooled_down_sec = time(0) - last_trigger_at;
    if (cooled_down_sec < cooldown_sec) {
      InferenceResultMsg msg;
      spdlog::info("Cooling down, {} seconds passed, {} seconds to go",
                   cooled_down_sec, cooldown_sec - cooled_down_sec);
      while (inference_result_pc_queue.try_dequeue(msg)) {
      }
      interruptible_sleep(10000, &ev_flag);
      continue;
    }
    spdlog::info("Handling batch");
    while (results.size() < batch_size && !ev_flag) {
      InferenceResultMsg msg;
      if (!inference_result_pc_queue.try_dequeue(msg)) {
        spdlog::warn("Unexpected result: "
                     "inference_result_pc_queue.try_dequeue() failed");
        break;
      }
      results.emplace_back(move(msg));
    }
    dequed_count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
      bool models_disagree = false;
      auto hist = vector<int>{0, 0, 0};
      for (auto const &lbl : results[i].labels()) {
        if (lbl >= hist.size()) {
          auto err_msg = fmt::format("unexpected label value {}", lbl);
          spdlog::error(err_msg);
          throw runtime_error(err_msg);
        }
        ++hist[lbl];
      }

      auto argmax = [](const auto &vec) -> size_t {
        /* if (vec.empty())
          throw std::runtime_error("Cannot find argmax of an empty vector");  */
        return std::distance(vec.begin(),
                             std::max_element(vec.begin(), vec.end()));
      };
      auto max_ele = argmax(hist);
      for (auto const &lbl : results[i].labels()) {
        if (max_ele != lbl) {
          models_disagree = true;
          break;
        }
      }
      labels[i] = max_ele;
      spdlog::info(
          "batch[{:2}], labels: {} ({}{:12}), image_size: {}K, image_ts: {}", i,
          max_ele, results[i].labels(), (models_disagree ? ", disagree" : ""),
          results[i].payload().length() / 1024,
          results[i].snapshotunixepochns() / 1000 / 1000);
    };
    int pre_detections = 5;
    for (size_t i = 0; i < results.size(); ++i) {
      if (labels[i] == 1) {
        ++dequed_count;
      } else
        break;
    }
    if (dequed_count > pre_detections) {
      spdlog::info("Removing {} out of {} items from the batch",
                   dequed_count - pre_detections, dequed_count);
      for (int i = 0; i < dequed_count - pre_detections; ++i) {
        results.pop_front();
      }
      continue;
    }

    neg_idx = -1;
    pos_idx = -1;

    auto contains = [](const vector<int> &vec, int element) {
      return find(vec.begin(), vec.end(), element) != vec.end();
    };
    for (size_t i = 0; i < batch_size; ++i) {
      if (contains(exclude_outputs, labels[i])) {
        neg_idx = i;
      }
      if (contains(include_outputs, labels[i])) {
        pos_idx = i;
      }
    }
    if (neg_idx == -1 && pos_idx >= 0) {
      prepare_images(results, labels);
      execute_external_program();
      last_trigger_at = std::time(0);
    }
    results.clear();
  }
  spdlog::info("inference_handling_ev_loop() exited gracefully");
}

void zeromq_consumer_ev_loop() {
  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ consumer connected to {}", zmq_address);
  constexpr int timeout = 5000;
  // For the use of zmqcpp, refer to this 3rd-party tutorial:
  // https://brettviren.github.io/cppzmq-tour/index.html#org3a79e67
  subscriber.set(zmq::sockopt::rcvtimeo, timeout);
  subscriber.set(zmq::sockopt::subscribe, "");

  while (!ev_flag) {
    zmq::message_t message;
    InferenceResultMsg msg;
    try {
      auto res = subscriber.recv(message, zmq::recv_flags::none);
      if (!res.has_value()) {
        spdlog::warn("subscriber.recv() timed out, will retry");
        continue;
      }
      if (res == 0) {
        spdlog::error("subscriber.recv() receives zero bytes, will retry");
        continue;
      }
    } catch (const zmq::error_t &e) {
      spdlog::error("subscriber.recv() throws exception: {}", e.what());
      continue;
    }
    if (message.size() == 0) {
      spdlog::warn("subscriber.recv() returned an empty message (probably due "
                   "to time out)");
      continue;
    }
    if (msg.ParseFromArray(message.data(), message.size())) {

      string labels = "";
      for (auto lbl : msg.labels()) {
        labels += to_string(lbl) + ", ";
      }
      /*
      spdlog::info(
          "recv()ed, labels: {}, image_size: {}, image_ts: {}, latency: {}ms",
          labels, msg.payload().length(),
          msg.snapshotunixepochns() / 1000 / 1000,
          (msg.inferenceunixepochns() - msg.snapshotunixepochns()) / 1000 /
              1000);
      */
      if (!inference_result_pc_queue.try_enqueue(msg)) {
        spdlog::warn("inference_result_pc_queue.try_enqueue(msg) failed, "
                     "consumer not dequeue()ing fast enough?");
      }
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_consumer_ev_loop() exited gracefully");
}

int main(int argc, char **argv) {
  install_signal_handler(&ev_flag);
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
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %5t | %v");
  spdlog::info("three_class_inference_consumer started (git commit: {})",
               GIT_COMMIT_HASH);
  config_path = result["config-path"].as<std::string>();
  ifstream f(config_path);
  json settings = json::parse(f);
  zmq_address =
      settings.value("/inference/zeromq_producer/address"_json_pointer,
                     "tcp://127.0.0.1:14241");
  external_program = settings.value(
      "/inference/on_detected/external_program"_json_pointer, "");
  gif_width =
      settings.value("/inference/on_detected/gif/size/width"_json_pointer, 100);
  gif_height = settings.value(
      "/inference/on_detected/gif/size/height"_json_pointer, 100);
  // The unit is "Time in 1/100ths of a second"
  gif_frame_interval =
      settings.value(
          "/inference/on_detected/gif/frame_interval_ms"_json_pointer, 200) /
      10;
  gif_path = settings.value("/inference/on_detected/gif/path"_json_pointer,
                            "/tmp/detected.gif");
  zmq_payload_mat_size = cv::Size(
      settings.value("/inference/zeromq/image_size/width"_json_pointer, 1920),
      settings.value("/inference/zeromq/image_size/height"_json_pointer, 1080));
  jpg_dir = settings.value(
      "/inference/on_detected/jpegs_directory"_json_pointer, "/");
  cooldown_sec =
      settings.value("/inference/on_detected/cooldown_sec"_json_pointer, 120);
  include_outputs = settings.value(
      "/inference/on_detected/triggers/include_outputs"_json_pointer,
      vector<int>{1});
  exclude_outputs = settings.value(
      "/inference/on_detected/triggers/exclude_outputs"_json_pointer,
      vector<int>{});
  thread thread_zeromq_consumer(zeromq_consumer_ev_loop);
  thread thread_inference_handling_ev_loop(inference_handling_ev_loop);
  if (thread_zeromq_consumer.joinable()) {
    thread_zeromq_consumer.join();
  }
  if (thread_inference_handling_ev_loop.joinable()) {
    thread_inference_handling_ev_loop.join();
  }
  spdlog::info("three_class_inference_consumer exited");
  return 0;
}
